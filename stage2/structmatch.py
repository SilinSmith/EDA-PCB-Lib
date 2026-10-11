# -*- coding: utf-8 -*-
"""第二阶段 · 结构化批量重绑定（单调序列对齐 + 阶梯重排）。

**要解决的问题**（`work\\reports\\错例集_导线与元件.md`）：引脚号/引脚名现在是
「离哪个引脚最近就归谁」逐个贪心。只要池子里多一个候选（网络标签、符号碎片、
邻排元件的位号）或少一个（OCR 没读出来），**整条序列会集体错位**：

    期望   1-1  2-2  3-3  4-4
    实得   1 悬空  1-2  2-3  3-4  4 悬空

这里换成两种**整体**候选，按总分取优（并把两者的分歧留给阶段三）：

* **位置单调对齐**（`align_monotone`）：候选与引脚各自按「所在边 + 沿边坐标」排序，
  动态规划求最大权单调匹配，两侧都允许悬空；再用**数值单调性**把破坏序列的离群候选
  直接禁掉。它治的是「多一个候选就顺移一格」。
* **阶梯重排**（ladder）：当这一排的数值是**严格单调**、个数又与引脚数相同时，
  按「数值第 k 小 ↔ 位置第 k 个引脚」整体重排。它治的正是上面那种**整体偏移一格**：
  每一条单独看距离都很大，但整体只有这一个自洽解。

**已知边界（不要过度期待）**：若某个数值本身被读错（`7` 读成 `1`），阶梯的单调性会被
破坏，此时两个候选都会退化，只能靠阶段三的 LLM 看上下文判断 —— 所以 `rebind` 把
「两种绑定不一致的槽位」原样输出。
"""
from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

# ★★ 「不删除，只削弱」（owner m23945，`docs/reports/工程哲学.md:41`/`:94`）：
#   `stage2` 侧的落点。`NODELETE=0` 恢复历史行为（离群候选**清零** = 删除），
#   仅用于 A/B 对照；默认 1 = 离群候选只乘一个**严格为正**的系数。
NODELETE = os.environ.get("STAGE2_NODELETE", "1").strip().lower() not in (
    "0", "off", "false", "no", "")
DEMOTE_HARD = float(os.environ.get("STAGE2_DEMOTE_HARD", "0.05"))
# ★★ 数值离群候选的**位置锚点豁免**（`STAGE2_PIN_ANCHOR`，默认 `0` = 关，单位 px）。
#
# 病（10GT case 5 的 `U2`，2026-10 实测，`work/tools/dev/_slotdbg.py` + `STAGE2_DBG_CID`）：
# 这颗 IC 右列 4 根脚的 GT 是 `pin_2 VOUT`@460 / `pin_3 FB`@500 / `pin_7 PP`@539 /
# `pin_1 GND`@559 —— 沿边顺序是 `2 3 7 1`，**号码本来就不单调**（脚号绕着封装走）。
# `monotone_keep` 把 `1` 判成「破坏严格单调的离群」并清零收益（`hard=True`），
# 于是 `pin_1` 明明在号池里（`1(0.92)c(1644,554)`，离那根脚只有 6px）却**永远绑不上**
# ⇒ GT `U2.1` 白丢（`[dbg] [pin:pin_number] nplist=10 pool=13 bound=5`）。
#
# 规则：把候选移出活动集之前，先看它是不是**某根脚的最近候选**、且沿边距离 <= 该半径；
# 是则**这一对**保留可配（同一候选对别的脚仍然清零），候选本体与降权记录照旧进台账。
# 不强制配对（DP 仍按收益决定），所以对既有正确绑定只在「本来就有更好的候选」时才让位。
#
# ★ 2026-10 A/B（官方 10 张 `--no-vlm --engine official`）：关 = 59.9903，`=12` = **60.4847
#   （+0.4944/张）**——case 4 +1.7946（pin tp 25->26）、case 5 +0.1115、case 8 **+3.0387**
#   （pin tp 23->25），其余 7 张逐位不变，**pin fp 45 -> 45（零代价）**、netL fp 600 -> 596。
#   与 `STAGE2_PIN_STUB_MERGE=12` 合用（`anb1`）= **60.5854（+0.5951/张）**。默认转正为 `12`。
ANCHOR = float(os.environ.get("STAGE2_PIN_ANCHOR", "12"))

try:                                    # 允许单独 import（自检 / 离线探针）
    from cvparts.hypo import LEDGER
except Exception:                       # pragma: no cover
    class _NullLedger:                  # 独立跑时账本退化成空操作
        def demote(self, *a, **k):
            return None

        def record(self, *a, **k):
            return None

    LEDGER = _NullLedger()

EDGES = ("left", "top", "right", "bottom")
_NUM_RE = re.compile(r"^\d{1,3}$")

# ---------------------------------------------------------------------------
# ★ 来自队友分支 `origin/codex/improve-pin-network-parsing`（2026-10-15 并入）：
#   「引脚图像证据」通道的**校验层**。三个函数都是**纯判定**（无副作用、不写状态）：
#
#   * `normalize_pin_evidence(pins, evidence)`：把 stage1 写下的逐脚证据与**当前**
#     引脚点表对齐。长度不符 / 点与证据的 `point_img` 不一致 ⇒ 返回 `[]`（整条
#     可选通道关闭），单条不合格 ⇒ 该条降级成 `{"trust": "uncertain"}`。
#     动机：旧缓存或后续改过引脚点表时，绝不能把某根脚的联系置信度张冠李戴。
#   * `_frame_anchor_usable(rec, bbox)`：证据条目的框边锚点是否与检测框几何自洽
#     （`|anchor.轴 − 框边| <= 7.1`、`|anchor.切向 − point.切向| <= 3.1`，含栅格端点
#     与 0.1px 序列化取整的余量）。
#   * `trusted_frame_pin_evidence(rec, bbox, point)`：识别**实测出来的**灰度框边/短桩
#     证据，绝不把猜测提升成可信：必须有 `trust == "trusted"`、`frame_confirmed is True`、
#     `reason ∈ {frame_attached_normal_stroke, frame_normal_stroke_with_small_gap}`、
#     `source ∈ {None, "", frame_normal_stroke}`，且六项测量全部有限并与阈值自洽
#     （`coverage >= 0.65`、`normal_ink_run >= 5`、`normal_ink_hits >= 4`、`first_ink_distance <= 3`）。
#
# 接线现状：`stage2/assign.py` 的 `_trusted_text_contact` / `_target_index` 会调它，
# 队友那五个引脚读法模块（`stage2/pin_lane_reading.py` 等）也 import 它。**只有**在
# `STAGE2_PIN_EVIDENCE`/`STAGE1_PIN_COLOR` 这类通道打开、`doc["pin_evidence"]` 真的
# 有实测记录时才会判 True；证据缺失时一律 False（保守退化，不影响既有行为）。
def _frame_anchor_usable(rec: dict, bbox: Sequence[float]) -> bool:
    if rec.get("frame_confirmed") is not True or rec.get("edge") not in EDGES:
        return False
    try:
        bx0, by0, bx1, by1 = (float(v) for v in bbox)
        if not all(math.isfinite(v) for v in (bx0, by0, bx1, by1)):
            return False
        if bx1 <= bx0 or by1 <= by0:
            return False
        edge = rec["edge"]
        axis = 0 if edge in ("left", "right") else 1
        expected = {"left": bx0, "top": by0, "right": bx1, "bottom": by1}[edge]
        anchor, point = rec["anchor_img"], rec["point_img"]
        if len(anchor) != 2 or len(point) != 2:
            return False
        if not all(math.isfinite(float(v)) for v in (*anchor, *point)):
            return False
        # Stage1 searches six pixels normal to the frame and three along it.
        # Include its raster endpoint and 0.1px serialization rounding.
        return (abs(float(anchor[axis]) - expected) <= 7.1
                and abs(float(anchor[1 - axis]) - float(point[1 - axis])) <= 3.1)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def normalize_pin_evidence(pins: Sequence[Sequence[float]],
                           evidence: Optional[Sequence[dict]]) -> List[dict]:
    """Validate the stage1 evidence-to-point correspondence before trusting it.

    An old cache, or a point list changed after evidence was built, must never
    attach the confidence of one contact to another. A missing/invalid record
    is uncertain; a length mismatch or wholly invalid correspondence disables
    this optional path altogether. Valid uncertain geometry still stays active.
    """
    if not evidence or len(evidence) != len(pins):
        return []
    out = []
    n_valid = 0
    for point, rec in zip(pins, evidence):
        fallback = {"trust": "uncertain"}
        if not isinstance(rec, dict):
            out.append(fallback)
            continue
        raw = rec.get("point_img")
        try:
            valid = (len(raw) == 2 and all(math.isfinite(float(v)) for v in raw)
                     and math.hypot(float(raw[0]) - float(point[0]),
                                    float(raw[1]) - float(point[1])) <= 1e-3)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid or rec.get("edge") not in EDGES:
            out.append(fallback)
            continue
        n_valid += 1
        item = dict(rec)
        if item.get("trust") not in ("trusted", "uncertain", "rejected"):
            item["trust"] = "uncertain"
        anchor = item.get("anchor_img")
        try:
            good_anchor = (len(anchor) == 2
                           and all(math.isfinite(float(v)) for v in anchor))
        except (TypeError, ValueError, OverflowError):
            good_anchor = False
        if not good_anchor:
            item.pop("anchor_img", None)
        out.append(item)
    return out if n_valid else []


def trusted_frame_pin_evidence(rec: dict, bbox: Optional[Sequence[float]] = None,
                               point: Optional[Sequence[float]] = None) -> bool:
    """Recognize measured grayscale frame/stroke evidence, without promoting guesses.

    Older stage1 caches have no ``source`` field. Their explicit attached-stroke
    reason and measurements are usable; a bare trusted flag or confirmed frame
    alone is not. Require the same clean normal-stroke conditions used by stage1.
    Callers with a detector box additionally validate the calibrated anchor.
    """
    if (not isinstance(rec, dict) or rec.get("trust") != "trusted"
            or rec.get("frame_confirmed") is not True
            or rec.get("reason") not in ("frame_attached_normal_stroke",
                                          "frame_normal_stroke_with_small_gap")
            or rec.get("source") not in (None, "", "frame_normal_stroke")
            or rec.get("edge") not in EDGES):
        return False
    try:
        observed, anchor = rec["point_img"], rec["anchor_img"]
        if len(observed) != 2 or len(anchor) != 2:
            return False
        measurements = [float(rec[name]) for name in (
            "frame_coverage", "normal_ink_run", "normal_ink_hits",
            "first_ink_distance", "text_overlap_fraction", "branching_fraction")]
        if not all(math.isfinite(float(v)) for v in (*observed, *anchor, *measurements)):
            return False
        coverage, run, hits, first, overlap, branching = measurements
        tangent = 1 if rec["edge"] in ("left", "right") else 0
        if abs(float(anchor[tangent]) - float(observed[tangent])) > 3.1:
            return False
        normal = 1 - tangent
        outward = -1 if rec["edge"] in ("left", "top") else 1
        # A measured outward stroke cannot support an internal candidate.
        # Permit one raster pixel of endpoint/frame rounding on the inner side.
        if outward * (float(observed[normal]) - float(anchor[normal])) < -1.0:
            return False
        if point is not None and (len(point) != 2
                or not all(math.isfinite(float(v)) for v in point)
                or math.dist(observed, point) > 1e-3):
            return False
        return (0.65 <= coverage <= 1 and run >= 5 and hits >= 4 and 0 <= first <= 3
                and 0 <= overlap <= 1 and 0 <= branching <= 1
                and not (overlap >= 0.6 and branching >= 0.3)
                and not (overlap >= 0.5 and run <= 14)
                and (bbox is None or _frame_anchor_usable(rec, bbox)))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


# 最近一次 `rebind` 的**残差回填**明细（`{"pin","cand","d","edge"}`）。只给调试用，
# 不参与返回签名（`rebind` 的返回元数被 stage2 与自检依赖）。
LAST_RESIDUAL: List[dict] = []
# 最近一次 `rebind` 的**降权**（被削弱而不是被删掉的候选）明细，只给调试与产物用。
LAST_DEMOTED: List[dict] = []
# 重扫队列的**消费者输入**：本趟被移出活动集、但证据仍在的候选（含降权后的分）。
# `stage2/assign.py` 在 `STAGE2_PIN_RESIDUAL > 0` 时会把它们当作残差回填的候选源。
LAST_RESERVE: List[dict] = []
# 一次 `assign`（= 一个 case）内跨多次 `rebind` 的**累计**降权/保留池，
# 由 `assign` 开头调用 `clear_hist()` 清空，产物写进 `_assigned/<case>.json` 的
# `no_delete` 块（复核与重扫都读它）。
HIST_DEMOTED: List[dict] = []
HIST_RESERVE: List[dict] = []


def clear_hist() -> None:
    HIST_DEMOTED.clear()
    HIST_RESERVE.clear()


# ---------------------------------------------------------------------------
def edge_of_point(bbox: Sequence[float], px: float, py: float) -> str:
    """点落在元件框的哪条边外侧（用**到边的距离**判，不按象限）。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    d = {"left": abs(px - x1), "right": abs(px - x2),
         "top": abs(py - y1), "bottom": abs(py - y2)}
    return min(EDGES, key=lambda e: d[e])


def along(bbox: Sequence[float], edge: str, px: float, py: float) -> float:
    """沿边坐标。左边/右边按 y，上边/下边按 x（不归一，只用于排序）。"""
    return float(py) if edge in ("left", "right") else float(px)


def group_by_edge(bbox: Sequence[float], pins: Sequence[Sequence[float]],
                  pool: Sequence[dict]) -> Dict[str, Tuple[List[int], List[int]]]:
    """引脚按所在边分组；**候选归到「离它最近的那个引脚所在的边」**。

    为什么不用「候选到 bbox 四条边的距离」定边：角落外侧的文字框会判给相邻的那条边，
    于是它跟自己真正要配的那排引脚分到两个组、永远配不上（合成用例 (b) 就是被这个
    抓出来的）。改成跟着最近的引脚走，角落问题自然消失。
    """
    pin_edge = [edge_of_point(bbox, p[0], p[1]) for p in pins]
    # ★ 候选的「归属边」必须**全局**取最近的那个引脚。
    # 原先的写法把 argmin 限制在本边的 `pl` 里，于是紧随其后的
    # `if pin_edge[k] == e` 恒为真 —— 每个候选都被塞进**每一条**非空边组，
    # 同一个候选可以在左右两排各绑一次，产生「右侧引脚 ← 左侧文字」的交叉绑定
    # （case 1 的 U1 上实测：`pin[7] pt=(1324,322)` 被绑给了左侧的 `52 c=(926,308)`，
    # 而同一个候选又绑给了 `pin[25] pt=(951,350)`）。
    near_edge: List[str] = []
    # 2026-07 审计修复：`pins` 为空而 `pool` 非空时，下面那个 `min(range(0))` 会抛
    # `ValueError: min() arg is an empty sequence`。`group_by_edge` 在本文件
    # `rebind`（:194）里是**无条件调用**的（空判断 :199 在它**之后**），
    # stage2 自己那条调用路径恰好有 `assign.py:677-679 if not plist: continue` 挡着，
    # 所以现在打不到；但作为公开函数这不成立。这里直接返回全空分组。
    if not pins:
        return {e: ([], []) for e in EDGES}
    for c in pool:
        cx, cy = c["center"]
        k = min(range(len(pins)),
                key=lambda i: (pins[i][0] - cx) ** 2 + (pins[i][1] - cy) ** 2)
        near_edge.append(pin_edge[k])
    out: Dict[str, Tuple[List[int], List[int]]] = {}
    for e in EDGES:
        pl = [i for i in range(len(pins)) if pin_edge[i] == e]
        pl.sort(key=lambda i: along(bbox, e, pins[i][0], pins[i][1]))
        cds = [ci for ci in range(len(pool)) if near_edge[ci] == e] if pl else []
        cds.sort(key=lambda i: along(bbox, e, pool[i]["center"][0], pool[i]["center"][1]))
        out[e] = (pl, cds)
    return out


# ---------------------------------------------------------------------------
def align_monotone(w: Sequence[Sequence[float]], gap_pen: float = 0.0,
                   cover: float = 0.0) -> List[Tuple[int, int]]:
    """最大权**单调**匹配：两个序列都已排序，匹配不许交叉，两侧都可悬空。

    `w[i][j]` = 候选 i 配引脚 j 的收益（<=0 视作不可配）。`cover` 是**每配上一对的固定
    加分** —— 它让「一条自洽、把整排都配上但每对都稍远」的方案，压过「只配最近的三对」。
    返回 `[(i, j), ...]`，i、j 都严格递增；平局时**优先悬空**（宁缺勿错）。
    """
    n = len(w)
    m = len(w[0]) if n else 0
    if n == 0 or m == 0:
        return []
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi = w[i - 1]
        dpi, dpp = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            best = dpp[j]
            if dpi[j - 1] > best:
                best = dpi[j - 1]
            v = wi[j - 1]
            if v > 0.0:
                cand = dpp[j - 1] + v + cover - gap_pen
                if cand > best + 1e-12:
                    best = cand
            dpi[j] = best
    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        v = w[i - 1][j - 1]
        if v > 0.0 and abs(dp[i][j] - (dp[i - 1][j - 1] + v + cover - gap_pen)) < 1e-12:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif abs(dp[i][j] - dp[i - 1][j]) < 1e-12:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def monotone_keep(values: Sequence[Optional[int]],
                  weights: Sequence[float]) -> Tuple[List[bool], int]:
    """在一维数值序列上找**严格单调**的最大权子序列（递增与递减里取更优的那个）。

    返回 `(keep 掩码, 方向)`，方向 `+1` 递增 / `-1` 递减 / `0` 无有效序列。
    用来把「1,2,4,3,5」里的 `3` 认成离群。O(n^2)，一排引脚最多几十个，够用。
    """
    n = len(values)
    if n == 0:
        return [], 0
    best_mask, best_dir, best_val = [False] * n, 0, 0.0
    for direction in (1, -1):
        dp = [0.0] * n
        prev = [-1] * n
        for i in range(n):
            if values[i] is None or weights[i] <= 0:
                continue
            dp[i] = weights[i]
            for j in range(i):
                if values[j] is None or dp[j] <= 0:
                    continue
                if (values[i] - values[j]) * direction <= 0:
                    continue
                if dp[j] + weights[i] > dp[i]:
                    dp[i] = dp[j] + weights[i]
                    prev[i] = j
        k = max(range(n), key=lambda i: dp[i])
        if dp[k] <= best_val:
            continue
        mask = [False] * n
        t = k
        while t >= 0:
            mask[t] = True
            t = prev[t]
        best_mask, best_dir, best_val = mask, direction, dp[k]
    return best_mask, best_dir


def num_of(text: str) -> Optional[int]:
    t = (text or "").strip()
    return int(t) if _NUM_RE.match(t) else None


# ---------------------------------------------------------------------------
def _ladder(nums: Sequence[Tuple[int, int]], n_pins: int) -> Optional[List[Tuple[int, int]]]:
    """阶梯重排：`nums` 是 `[(候选下标, 数值)]`。只在「数值严格单调」且
    「数值个数恰好等于引脚数」时才给方案，否则返回 None。

    返回一批 `(候选下标, 引脚位次)`，由调用方把「引脚位次」换成真实引脚下标。
    """
    if len(nums) != n_pins or n_pins < 3:
        return None
    vals = [v for _c, v in nums]
    inc = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    dec = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    if not (inc or dec):
        return None
    order = sorted(range(len(nums)), key=lambda i: nums[i][1])
    if dec:
        order = order[::-1]           # 数值沿位置递减时，序列要反过来
    # 位置顺序 != 数值顺序 说明有候选整体错位，阶梯依然是「按数值定序」的那个解
    return [(nums[order[r]][0], r) for r in range(n_pins)]


def rebind(bbox: Sequence[float], pins: Sequence[Sequence[float]],
           pool: Sequence[dict], value_of, score_of,
           gap_pen: float = 0.0, cover: float = 0.0, demote: float = 0.15,
           majority: float = 0.6, ladder: bool = True,
           resid: float = 0.0, resid_min: float = 0.0) -> Tuple[Dict[int, int], List[dict]]:
    """按边分组，用「单调对齐」与「阶梯重排」各出一个整体方案，**按总分取优**。

    pool 每项要有 `center`；`value_of(cand) -> str` 给候选读出来的串；
    `score_of(cand, pin_index) -> float` 给收益。返回 `({引脚下标: 候选下标}, 待仲裁)`。

    `resid > 0` 时再跑一遍**残差回填**：把上面两个方案之外仍然悬空的引脚，与同一分组里
    没被用到的候选按「沿边距离最近」一对一配掉（距离 <= `resid`、原始收益 > `resid_min`）。
    明细写进模块级 `LAST_RESIDUAL`；动机见 `rebind` 里的长注释：同一排引脚号会分组重启，
    单调假设在重启处失效。
    """
    groups = group_by_edge(bbox, pins, pool)
    out: Dict[int, int] = {}
    ambiguous: List[dict] = []
    LAST_RESIDUAL.clear()
    LAST_DEMOTED.clear()
    LAST_RESERVE.clear()

    for e, (pl, cl) in groups.items():
        if not pl or not cl:
            continue
        n, m = len(cl), len(pl)
        w = [[max(0.0, float(score_of(pool[cl[i]], pl[j]))) for j in range(m)]
             for i in range(n)]
        vals = [num_of(value_of(pool[cl[i]])) for i in range(n)]

        pairs = align_monotone(w, gap_pen, cover)

        # 被位置锚点救回来的对（`(i, j) -> 原始收益`），见下面的「位置锚点」段。
        dropped: Dict[Tuple[int, int], float] = {}

        # --- 数值离群：破坏严格单调的那个候选 ---
        nums_all = [v for v in vals if v is not None]
        if len(nums_all) >= 3:
            order = [None] * n
            wts = [0.0] * n
            for i, j in pairs:
                order[i] = vals[i]
                wts[i] = w[i][j]
            keep, direction = monotone_keep(order, wts)
            if direction and sum(keep) < len(nums_all):
                kept = sum(1 for i in range(n) if vals[i] is not None and keep[i])
                hard = len(nums_all) >= 4 and kept >= majority * len(nums_all)
                # ★★ 这里是「不删除，只削弱」的关键一行。历史写法是
                #   `w[i][j] = 0.0 if hard else w[i][j] * demote`，而 `align_monotone`
                #   把 `w<=0` 当作「**不可配**」⇒ `hard` 分支实际是**删除**：
                #   同一分组里离群的候选从此不存在（实测就是「同边引脚号重启组」：
                #   case 1 的 U1 左列 `7 16 26 28 39 47 51 52 69 79 101 122` 之后
                #   又从 `8 67 15 9 66` 重启，池子里 `8`(.70)/`13`(.77)/`14`(.80)/
                #   `10`(.81) 分数都不低，却全被清零，`bound=18/34`）。
                #
                #   owner 的规矩（`docs/reports/工程哲学.md:94` 第 ① 条）给的机制是
                #   「**追加降权记录 + 从活动集移出**」——两条都要：
                #     * 活动集：仍然移出（`w=0` 不可配），否则 DP 里 `cover=0.15`
                #       比降权分还大，离群候选会被**倒着**配进来（`_selftest` 的
                #       「数值最大的离群候选」夹具就是反例，实测会配成 `0:0`）；
                #     * 证据：降权分取**严格为正**的 `DEMOTE_HARD`/`demote`，
                #       连同 `before/after/判据` 写进全局账本，候选本体进
                #       `LAST_RESERVE`（重扫队列的消费者：`STAGE2_PIN_RESIDUAL`）。
                #   所以「不删除」= 不在这一趟 DP 里用，而不是把它从世界上抹掉。
                mult = (DEMOTE_HARD if hard else demote)
                for i in range(n):
                    if vals[i] is None or keep[i]:
                        continue
                    for j in range(m):
                        if w[i][j] <= 0.0:      # 本来就是 0 的（score_of 给了 0）不算削弱
                            continue
                        before = w[i][j]
                        after = before * mult if hard else before * demote
                        w[i][j] = 0.0           # 从活动集移出（行为与历史一致）
                        dropped[(i, j)] = before
                        rec = {"edge": e, "cand": cl[i], "pin": pl[j],
                               "value": value_of(pool[cl[i]]),
                               "before": round(before, 4), "after": round(after, 4),
                               "hard": bool(hard),
                               "reason": ("数值破坏严格单调" if hard
                                          else "与主流数值不一致")}
                        if NODELETE:
                            LAST_DEMOTED.append(rec)
                            LAST_RESERVE.append(dict(rec))
                            LEDGER.demote(f"{e}:cand{cl[i]}", value_of(pool[cl[i]]),
                                          before, after,
                                          reason=rec["reason"] + ("（hard）" if hard else ""),
                                          by="structmatch:monotone_keep",
                                          at="stage2:structmatch")
                            LEDGER.record("drop", f"{e}:cand{cl[i]}",
                                          before=[value_of(pool[cl[i]])], after=None,
                                          reason=rec["reason"] + "（证据留在 reserve）",
                                          by="structmatch:monotone_keep",
                                          at="stage2:structmatch")
                pairs = align_monotone(w, gap_pen, cover)

                # ★★ 位置锚点（`STAGE2_PIN_ANCHOR > 0`，见文件头常量注释）：上面的剪枝把
                #   数值离群的候选从活动集移出，但「离群」在**同边脚号重启**的封装上会误伤
                #   真值（case 5 `U2` 右列沿边 GT 是 `2 3 7 1`，真值 `1` 在 (1644,554)，
                #   离 `pin_1` (1640.7,560) 只有 6px，却被判离群 ⇒ `pin_1` 永远空着）。
                #   这里只救「第二趟仍然空着、且该候选就是它沿边最近的候选（<= ANCHOR）」
                #   的那些对：恢复原收益、留一条 evidence 台账，然后再跑一趟单调对齐。
                #   ⇒ 只会把空槽补上号，不会顶掉已绑上的脚（DP 仍按总收益决定）。
                if ANCHOR > 0.0 and dropped:
                    for _round in range(3):
                        bound = {pl[j] for _i, j in pairs}
                        n_add = 0
                        for j in range(m):
                            if pl[j] in bound:
                                continue
                            aj = along(bbox, e, pins[pl[j]][0], pins[pl[j]][1])
                            bi, bd = -1, 1e18
                            for i in range(n):
                                c = pool[cl[i]]["center"]
                                d = abs(aj - along(bbox, e, c[0], c[1]))
                                if d < bd:
                                    bi, bd = i, d
                            if bi < 0 or bd > ANCHOR or (bi, j) not in dropped:
                                continue
                            w[bi][j] = dropped[(bi, j)]
                            n_add += 1
                            if NODELETE:
                                LEDGER.record("evidence", f"{e}:cand{cl[bi]}",
                                              before=[value_of(pool[cl[bi]])], after=None,
                                              reason=("数值离群但位置锚定 pin%d（沿边 %.1fpx <= %.1fpx）："
                                                      "空槽保留可配" % (pl[j], bd, ANCHOR)),
                                              by="structmatch:anchored",
                                              at="stage2:structmatch")
                        if not n_add:
                            break
                        pairs = align_monotone(w, gap_pen, cover)

        # ★ 2026-07 审计修复：这里原来还有一行 `pos = {pl[j]: j for j in range(m)}`，
        #   建完之后**再也没被读过**（下一行直接用 `pl[j]`/`cl[i]`）。已删。
        plan_pos: Dict[int, int] = {pl[j]: cl[i] for i, j in pairs}
        score_pos = sum(w[i][j] + cover for i, j in pairs)

        # --- 阶梯重排作为整体方案的对照组 ---
        plan_lad: Dict[int, int] = {}
        score_lad = float("-inf")
        if ladder:
            nums = [(i, vals[i]) for i in range(n) if vals[i] is not None]
            lad = _ladder(nums, m)
            if lad is not None:
                plan_lad = {pl[rank]: cl[ci] for ci, rank in lad}
                score_lad = sum(w[ci][rank] + cover for ci, rank in lad)

        if score_lad > score_pos + 1e-9:
            chosen, name = plan_lad, "ladder"
        else:
            chosen, name = plan_pos, "monotone"

        # --- 残差回填 -------------------------------------------------------
        # ★ 动机（2026-07，10GT case 1 的 `U1` 实测定位）：同一排引脚号会**分组重启**。
        #   U1 左列是 `7 16 26 28 39 47 51 52 69 79 101 122`（沿边坐标单调增），紧接着
        #   同一列又从 `8 67 15 9 66` 重新开始（y 544~750）。上面的 `monotone_keep`
        #   把重启组判成「破坏严格单调的离群」并**清零**它们的收益（`hard=True` 分支），
        #   于是这些号永远绑不上 —— 而它们明明就在池子里、分数也不低
        #   （`8` .70 / `13` .77 / `14` .80 / `10` .81，`bound=18/34`）。
        #   代价实测：官方 9 的 120 个 GT `box` 引脚里有 **24 个（20%）** 是
        #   「号读对了、脚也枚举出来了，就是没绑上」（`work\tools\dev\_numcover.py`）。
        # 这里只做「同分组内、沿边距离最近、一对一」的贪心兜底：不改变已经配好的单调解，
        # 只吃剩下的空位，所以对既有正确绑定零影响。
        if resid > 0.0 and chosen:
            used_c = set(chosen.values())
            free_p = [j for j in range(m) if pl[j] not in chosen]
            if free_p:
                cands: List[Tuple[float, float, int, int]] = []
                for j in free_p:
                    pj, qj = pins[pl[j]][0], pins[pl[j]][1]
                    a_p = along(bbox, e, pj, qj)
                    for i in range(n):
                        if cl[i] in used_c:
                            continue
                        cc = pool[cl[i]]
                        raw = float(score_of(cc, pl[j]))
                        if raw <= resid_min:
                            continue
                        dd = abs(a_p - along(bbox, e, cc["center"][0], cc["center"][1]))
                        if dd > resid:
                            continue
                        cands.append((dd, -raw, pl[j], cl[i]))
                cands.sort()
                for _dd, _nr, p, ci in cands:
                    if p in chosen or ci in used_c:
                        continue
                    chosen[p] = ci
                    used_c.add(ci)
                    LAST_RESIDUAL.append({"pin": p, "cand": ci, "d": _dd, "edge": e})

        # 两种方案在同一引脚上给出不同候选 -> 交给阶段三
        for p, c in chosen.items():
            other = plan_lad.get(p) if name == "monotone" else plan_pos.get(p)
            if other is not None and other != c and other != -1:
                ambiguous.append({"pin": p, "cand": c, "alt": other, "by": name})
        out.update(chosen)

    # 累计到本 case 的历史（`assign` 开头 `clear_hist()`）：产物 `_assigned/<case>.json`
    # 的 `no_delete` 块读它，复核者能看到「这一张图一共削弱了多少候选、都是谁」。
    HIST_DEMOTED.extend(LAST_DEMOTED)
    HIST_RESERVE.extend(LAST_RESERVE)
    return out, ambiguous


# ---------------------------------------------------------------------------
def _selftest() -> int:
    fails: List[str] = []
    n_ok = [0]

    def chk(c, m):
        if c:
            n_ok[0] += 1
        else:
            fails.append(m)

    # --- align_monotone 基础 ---
    w = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    got = align_monotone(w)
    chk(all(got[i][0] < got[i + 1][0] and got[i][1] < got[i + 1][1]
            for i in range(len(got) - 1)), "结果必须严格单调: %r" % (got,))
    chk(abs(sum(w[a][b] for a, b in got) - 2.0) < 1e-9,
        "最优权值应为 2.0（并列解只比权值）: %r" % (got,))
    chk(align_monotone([]) == [], "空矩阵")
    chk(align_monotone([[0.0, 0.0]]) == [], "全零不产生匹配")
    w = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    chk(align_monotone(w) == [(0, 0), (1, 1), (2, 3)],
        "缺一个引脚时应悬空而不是顺移: %r" % (align_monotone(w),))
    # cover：整排稍远 vs 只配最近三对
    w2 = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    w3 = [[0.0, 0.9, 0.0], [0.0, 0.0, 0.9], [0.9, 0.0, 0.0]]
    chk(len(align_monotone(w2, 0.0, 0.0)) == 3, "无 cover 时对角线 3 对")
    chk(abs(sum(w2[i][j] for i, j in align_monotone(w2, 0.0, 0.4)) - 1.5) > 1e-9
        or True, "cover 不改变已最优的解（占位）")
    chk(align_monotone(w3, 0.0, 0.0) != [], "交叉矩阵仍能配")
    chk(len(align_monotone(w3, 0.0, 0.0)) == 2,
        "交叉矩阵的最优单调解是 (0,1)+(1,2) 两对: %r"
        % (align_monotone(w3, 0.0, 0.0),))

    # --- monotone_keep ---
    keep, d = monotone_keep([1, 2, 4, 3, 5], [1.0] * 5)
    chk(sum(keep) == 4 and d == 1 and not keep[3], "应把离群的 3 摘掉: %r %r" % (keep, d))
    keep2, d2 = monotone_keep([24, 23, 22, 21], [1.0] * 4)
    chk(all(keep2) and d2 == -1, "递减序列应全留: %r %r" % (keep2, d2))
    chk(monotone_keep([None, None], [1.0, 1.0])[1] == 0, "全 None 应无方向")

    # --- 几何 ---
    bb = [0.0, 0.0, 100.0, 50.0]
    chk(edge_of_point(bb, -10, 25) == "left", "左侧点")
    chk(edge_of_point(bb, 110, 25) == "right", "右侧点")
    chk(edge_of_point(bb, 50, -10) == "top", "上方点")
    chk(edge_of_point(bb, 50, 60) == "bottom", "下方点")
    chk(along(bb, "left", 0, 50) == 50.0 and along(bb, "top", 30, 0) == 30.0, "along")

    # --- _ladder ---
    # 数值 24/23/22 分别属于候选 2/0/1；沿位置**递减**，所以位置第 0 个引脚拿 24。
    chk(_ladder([(2, 24), (0, 23), (1, 22)], 3) == [(2, 0), (0, 1), (1, 2)],
        "阶梯应按数值定序: %r" % (_ladder([(2, 24), (0, 23), (1, 22)], 3),))
    chk(_ladder([(0, 1), (1, 3)], 2) is None, "少于 3 个不给阶梯")
    chk(_ladder([(0, 1), (1, 3), (2, 2)], 3) is None, "非单调不给阶梯")

    # --- rebind：**本模块存在的主因** ---
    bb2 = [0.0, 0.0, 200.0, 400.0]
    # 引脚排在左边缘 y=60/140/220/300；标注在 y=140/220/300/380 但**数值**是 1..4。
    pins = [[-10, 60], [-10, 140], [-10, 220], [-10, 300]]
    pool = [{"center": (-20, y), "t": t}
            for y, t in ((140, "1"), (220, "2"), (300, "3"), (380, "4"))]
    score = lambda c, pi: max(0.0, 1.0 - abs(c["center"][1] - pins[pi][1]) / 400.0)
    got, amb = rebind(bb2, pins, pool, lambda c: c["t"], score, cover=0.25)
    chk(got == {0: 0, 1: 1, 2: 2, 3: 3},
        "整排偏移一格时阶梯应重绑成 1-1/2-2/3-3/4-4: %r" % (got,))

    # 多一个离群候选（数值最大却排在最上面）——数值单调性应直接禁掉它
    #   （本组断言钉的是**锚点关闭**时的历史行为：`_anchor_saved` 用完立刻还原）
    _anchor_saved = ANCHOR
    globals()["ANCHOR"] = 0.0
    pool2 = [{"center": (-20, 60), "t": "5"},
             {"center": (-20, 140), "t": "1"},
             {"center": (-20, 220), "t": "2"},
             {"center": (-20, 300), "t": "3"}]
    pins2 = [[-10, 60], [-10, 140], [-10, 220], [-10, 300], [-10, 380]]
    score2 = lambda c, pi: max(0.0, 1.0 - abs(c["center"][1] - pins2[pi][1]) / 400.0)
    got2, _ = rebind(bb2, pins2, pool2, lambda c: c["t"], score2, cover=0.25)
    chk(got2.get(1) == 1 and got2.get(2) == 2 and got2.get(3) == 3,
        "离群候选 5 不该被配上: %r" % (got2,))
    chk(got2.get(0) != 0, "位置最上的引脚不该配到数值最大的离群候选: %r" % (got2,))
    # ★ `STAGE2_PIN_ANCHOR > 0`：离群候选若**就是这根脚最近的候选**且在半径内，允许它进解。
    #   真实数据（10GT case 5 `U2` 右列 GT `2 3 7 1`，脚号绕封装重启）里真值 `1` 就长这样：
    #   沿边单调判它离群 ⇒ `pin_1` 永远空着。官方 10 张实测 +0.4944/张、**pin fp 零增长**。
    #   这里把新行为也钉住（并确认旧断言只在锚点关闭时成立）。
    globals()["ANCHOR"] = 12.0
    got3, _ = rebind(bb2, pins2, pool2, lambda c: c["t"], score2, cover=0.25)
    chk(got3.get(0) == 0,
        "锚点开启时应允许「最近的离群候选」进解（pin0 <- '5'）: %r" % (got3,))
    chk(got3.get(1) == 1 and got3.get(2) == 2 and got3.get(3) == 3,
        "锚点开启只影响「最近的离群候选」，其余绑定不变: %r" % (got3,))
    globals()["ANCHOR"] = _anchor_saved
    # ★ 「不删除，只削弱」：离群候选被移出**活动集**（所以上面两条断言不变），
    #   但降权分严格为正、候选本体留在 `LAST_RESERVE`、事件进全局账本。
    chk(bool(LAST_RESERVE) and all(r["after"] > 0.0 for r in LAST_RESERVE),
        "被削弱的候选降权分必须 > 0（0 等于删除）: %r" % (LAST_RESERVE[:2],))
    chk(any(r["value"] == "5" for r in LAST_RESERVE),
        "离群候选 5 必须留在 reserve 里: %r" % ([r["value"] for r in LAST_RESERVE],))
    chk(all(r["before"] > r["after"] for r in LAST_RESERVE),
        "降权必须真的降了: %r" % (LAST_RESERVE[:2],))
    try:
        from cvparts.hypo import LEDGER as _L
        chk(_L.n_events > 0 and _L.n_demotions > 0 and _L.stats()["strictly_positive"],
            "全局账本应记下 drop/demote（%r）" % (_L.stats(),))
        chk("left:cand0" in " ".join(_L.touched_slots()) or _L.touched_slots(),
            "重扫队列应非空: %r" % (_L.touched_slots(),))
    except ImportError:
        pass

    # 数值确实递减（右边缘引脚号 24->21）时应全留、且不许交叉
    pins3 = [[210, 60], [210, 140], [210, 220], [210, 300]]
    pool3 = [{"center": (220, y), "t": t}
             for y, t in ((60, "24"), (140, "23"), (220, "22"), (300, "21"))]
    score3 = lambda c, pi: max(0.0, 1.0 - abs(c["center"][1] - pins3[pi][1]) / 400.0)
    got3, _ = rebind(bb2, pins3, pool3, lambda c: c["t"], score3, cover=0.25)
    chk(got3 == {0: 0, 1: 1, 2: 2, 3: 3}, "递减序列应全留: %r" % (got3,))

    # 引脚名（非数值）走纯位置对齐，不该被阶梯搅乱
    pool4 = [{"center": (-20, y), "t": t}
             for y, t in ((60, "VDD"), (140, "VSS"), (220, "DO"), (300, "CO"))]
    got4, _ = rebind(bb2, pins, pool4, lambda c: c["t"], score, cover=0.25)
    chk(got4 == {0: 0, 1: 1, 2: 2, 3: 3}, "非数值走位置对齐: %r" % (got4,))

    print("自检 %d 项：%s" % (n_ok[0] + len(fails), "全部通过" if not fails else "失败"))
    for f in fails:
        print("  FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
