# -*- coding: utf-8 -*-
"""全局引脚号 / 名一对一指派（开关 `STAGE2_PIN_SOLVE`，默认 **关**）。

## 病

`assign.assign()` 的引脚绑定是**逐角色、逐槽**的：

  * 名轮（`assign.py:2453-2455` 的 `("pin_name", ...)`）先跑，把框内的文字按
    `structmatch.rebind()` 单调对齐到整排引脚上，绑完就 `taken_box.add()`；
  * 号轮再跑，只能拿名轮**剩下**的框；
  * 最后 `PINNAME_OUTSIDE` 的框外补名（`assign.py:2681`）再补一遍空槽。

每一轮都是「一个角色 × 一排槽」的**局部**决策，于是有三类结构性问题：

  1. 号与名来自**不同**候选（号在框外、名在框内，是这本数据的物理常态）时，
     两轮各自独立最优并不等于**联合**最优 —— 号轮选走的框，可能是名轮更需要的；
  2. `structmatch.rebind()` 的单调性假设在「触点比真脚多」（stage1 的幻影触点）
     时会把整排号码错位一格，而逐槽路径**没有回退机制**；
  3. 同一块文字被两个角色轮分别看中时，先到先得，而不是「谁的总增益大给谁」。

## 做法

对每颗 `type in PIN_TYPES` 的元件，在**它所有槽**（`pins` 的每个 slot）与
**它附近所有候选读数**（沿用 `assign()` 已经算好的 `cands` / `steal` 记录，
不重新 OCR）之间做一次**最大权一对一指派**：

    行 = (槽 i, 字段 number|name)     共 2N 行
    列 = 一个候选文字框              共 M 列（+ 每行一个 0 权哑列 = 「不绑」）
    权 = `assign._score()` 的既有加权和（置信度 / fit / 距离 / 框内加成，系数一个不新造）
         **加上槽相关项** `assign._score(0.0, 0.0, dp, False)`（`dp` = 候选中心到该槽点
         的距离）—— 这一项不是新东西，就是 `_score` 自己的 `dist_w`，也是
         `assign.py:2717-2719` 框外补名手写的那一条 `- 0.035 * dp`。

⚠ **槽相关项是必需的**，不是锦上添花：`_score` 的 `d` 是「文字框到**元件框**」的距离，
和槽无关，所以只用 `_score` 时同一个候选摆在每个槽上权重都相同 —— 目标函数退化成
「尽量多绑不同的候选」，**谁摆在哪一槽**由匈牙利任意破平。实测（13 页，2026-11-02）
自由重排下 pin f1 从 0.7857 掉到 0.0361：case 3 的 `U1` 六个号被整体置换
（`1/BS` 变 `2/EN`），而总分还"变高了"。加入 `dp` 项后槽的位置才有意义。

一个候选只能占一列 ⇒ **一对一**；同一槽的 number 与 name 是**两行**，可以来自
不同的候选 ⇒ 这就是「联合增益」。约束（号必须过 `PINNUM_RE`、名/号各占一个
字段）全部由调用方在造权重时套用既有闸门，本模块不重复判定。

## 保守闸门

`solve()` 返回的是**最终**指派与它的总增益：

  * 求解结果里「没被指派」的字段，回填逐槽贪心原来的绑定（只要该候选没被
    别处用掉）⇒ **只增不减**，不会因为求解把一个字段删掉；
  * 最终总权 **不大于** 逐槽贪心原来的总权时，调用方原样保留贪心结果
    （平局也保留），所以本臂只可能在「联合增益 > 0」时改变输出。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# 判题口径的「有脚」元件类型，与 `schematic_parser/data.py:IC_TYPES` 同一个集合
# （`schematic_parser/metrics.py:460` 的 `pin_f1` 用它决定只数哪些件）。
PIN_TYPES = ("box", "ic", "boxic")

FIELD_ROLE = ("number", "name")      # 行号 0/1 对应的 `pins[i]` 字段名
NUM, NAME = 0, 1


def _hungarian(cost: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    """最小权匹配（e-maxx 版匈牙利，`O(n^2 m)`，要求行数 <= 列数）。

    返回 `[(行, 列), ...]`，每行恰好一个列。数值规模在这里很小（行 <= 2*脚数），
    纯 Python 足够快；不引第三方求解器，避免给管线加依赖。
    """
    n = len(cost)
    if not n:
        return []
    m = len(cost[0])
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            row = cost[i0 - 1]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


def solve(n_slots: int, weights: Dict[Tuple[int, int], Dict[int, float]],
          cur: Dict[Tuple[int, int], int],
          pinned: Iterable[Tuple[int, int]] = (),
          eps: float = 1e-9) -> Tuple[Dict[Tuple[int, int], int], float]:
    """最大权一对一指派 + 「不删」回填。

    `weights[(slot, f)][ci]` = 该候选摆在该槽上的权（调用方用 `assign._score()` 的
        既有式子造好，含**槽相关**的距离项）；`<= 0` 视为不可用 —— 与逐槽贪心的
        `if s <= 0: continue` 同一条线（`assign.py:2518`）。
    `cur[(slot, f)]` = 逐槽贪心当前绑定的候选下标（同时是总权比较的基线）。
    `pinned` = 不允许改动的行（保守档：既有绑定一个不动，求解器只填空字段）。
        这些行**不进代价矩阵**、它们的候选列也被保留掉 —— 于是「只填空字段」是
        硬保证，不依赖权重正负：一个已经绑好的槽既不会被换、也不会被清。
        `pinned` 里不在 `cur` 的 key 同样被排除（那是有值但认不出候选的字段）。

    返回 `(最终指派 {(slot, f): ci}, 增益)`；增益 `<= 0` 时调用方应原样保留。
    """
    pinned = set(pinned)
    pinned_bind = {k: cur[k] for k in pinned if k in cur}
    reserved = set(pinned_bind.values())
    rows: List[Tuple[Tuple[int, int], Dict[int, float]]] = []
    for i in range(n_slots):
        for f in (NUM, NAME):
            key = (i, f)
            if key in pinned:
                continue
            ws: Dict[int, float] = {ci: float(w)
                                    for ci, w in (weights.get(key) or {}).items()
                                    if w > 0.0 and ci not in reserved}
            if ws or key in cur:
                rows.append((key, ws))
    got: Dict[Tuple[int, int], int] = dict(pinned_bind)
    if not rows:
        return got, 0.0

    cols = sorted({ci for _k, ws in rows for ci in ws}
                  | {ci for ci in cur.values()})
    col_of = {ci: j for j, ci in enumerate(cols)}
    n_rows, n_cols = len(rows), len(cols) + len(rows)     # 每行一个 0 权哑列
    cost = []
    for key, ws in rows:
        line = [0.0] * n_cols
        for ci, w in ws.items():
            line[col_of[ci]] = -w
        cost.append(line)

    weight_of = {(key, ci): w for key, ws in rows for ci, w in ws.items()}
    used = set(reserved)
    for r, c in _hungarian(cost):
        if c >= len(cols):
            continue                                    # 哑列 = 该行不绑
        key = rows[r][0]
        ci = cols[c]
        if weight_of.get((key, ci), 0.0) > 0.0:
            got[key] = ci
            used.add(ci)

    total = sum(weight_of.get((k, ci), 0.0) for k, ci in got.items())
    for key, ci in cur.items():                         # 只增不减：回填未被占用的旧绑定
        if key in got or ci in used:
            continue
        got[key] = ci
        used.add(ci)
        total += weight_of.get((key, ci), 0.0)
    base = sum(weight_of.get((k, ci), 0.0) for k, ci in cur.items())
    return got, total - base
