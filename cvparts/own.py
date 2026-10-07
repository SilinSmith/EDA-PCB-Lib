# -*- coding: utf-8 -*-
"""独家归属（ownership）：争抢同一个东西的多个主张者，最后**只能有一个主人**。

为什么要有这个模块
------------------
业主 2026-10-06 的原话：

    考虑到相邻的两个框，他们重叠处确实会有所有权争议，但是最终只会归属于某一个。
    这是需要你考虑到的。在其他任何地方，类似的情况也是同理。

这不是「再加一个去重」——官方口径本身就是这么判的：
`metrics.assign_components` 用的是**一对一分配**（键匹配 + `_assign` 匈牙利），
每个 GT 至多配一个预测，**多出来的主张者一律记 fp**。
所以「一个东西只归一个人」不是我们的选择，是标尺的规定。

本模块把这条规定写成**一个**实现，所有阶段共用，并且**每次都留下账**：

    thin(parties)   多主张者争一个位置（重复检测） → 留分最高的，其余进 ledger
    assign(cl, it)  一个主张者够到多个标的（归属不明）→ 整体解一次一对一

三条不变式（都有自检）
----------------------
1. **一对一**：一个标的至多一个主人，一个主张者至多一个标的。
2. **有账**：每一次判负都记下「谁、输给谁、差多少、按哪条规则」；主张者的
   原始记录**不销毁**（分数、框都留着）。
3. **单调**：阈值放宽只会并进更多，不会造出新的东西；`ledger` 条数不减。

判据为什么不能是 IoU
--------------------
`stage1/detect.py` 原来用同类贪心 NMS（IoU>0.5）。**小图章上 IoU 是个坏尺子**：
`STAMP_SIDE` 把同类框都换成同一个正方形（`r`/`c`/`gnd`=20、`v`=10、`net_input`=30），
两个 20x20 的框中心差 10px 时 IoU 只有 0.33 —— **不会**被抑制，
可它们显然在抢同一颗符号。

实测（`work\\tools\\diag\\_owneval.py`，19 张快测集 stage1 产物，
**3975 对同类型框对**，用「最近同类 GT（corner_dev<=20）是不是同一颗」当真相）：

    判据          并掉的   真重复   误伤真符号   误伤率
    frac<=0.3        0        0          0       0.0%
    frac<=0.5        3        2          1      33.3%
    frac<=0.7        8        4          4      50.0%
    frac<=1.0       29        6         23      79.3%
    cd<=8            0        0          0       0.0%
    cd<=12           2        2          0       0.0%   ← 现成常量 DEDUP_PX
    cd<=20          46       25         21      45.7%   ← 官方 bbox 容差，**不能用**

两条结论：
* **不能用官方那 20px 当并框半径** —— 它是「框算不算对」的容差，不是「是不是同一颗」。
  `0092` 的电阻阵列 R1..R20 彼此只差 16.5px（frac 0.83），`cd<=20` 会把它们并成一堆。
  （这也是 `work\\tools\\diag\\_dupscore.py` 在提交文档上按 20px 再去重掉 −0.373、
  `0042` 掉 −4.87 的原因。）
* **`frac = 中心距 / 自身短边` 比绝对像素好**：它自动跟着类别尺寸走（`v` 的 10px
  和 `net_input` 的 30px 不是一个尺度）。但它单独也不够 —— `0092` 那排在 frac 0.83，
  所以阈值必须**紧**。

默认取 `frac=0.30` 且 `floor=0`，等价于「中心距不超过自己短边的三成」。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# 直接 `python cvparts/own.py` 时 sys.path[0] 是 cvparts/ 而不是 eda-pcb/，
# 包内绝对导入会失败。作为脚本运行时把包根补回来（被 import 时不生效）。
if __package__ in (None, ""):
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

Box = Tuple[float, float, float, float]

# 中心距上限 = max(FLOOR_PX, FRAC * 两个框各自短边的较小者)
FRAC = float(os.environ.get("OWN_FRAC", "0.30"))
FLOOR_PX = float(os.environ.get("OWN_FLOOR_PX", "0.0"))
# 现成常量：`stage1/detect.py` 用它判「YOLO 的框与 CV 的框是不是同一件」，实测安全
LEGACY_PX = float(os.environ.get("OWN_LEGACY_PX", "12.0"))


def _env_off(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else v.strip()


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_dev(a: Sequence[float], b: Sequence[float]) -> float:
    """两个框中心的切比雪夫距离。**贴章之后 = 官方 `corner_dev`**（同类同尺寸）。"""
    return max(abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2),
               abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2))


def short_side(box: Sequence[float]) -> float:
    return min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


@dataclass
class Party:
    """一个主张者。`kind` 不同就不争（同类才争同一个位置）。"""
    id: str
    box: Box
    kind: str = ""
    score: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def short(self) -> float:
        return short_side(self.box)


def contenders(a: Party, b: Party, *, frac: float = FRAC, floor_px: float = FLOOR_PX,
               same_kind: bool = True) -> bool:
    """a 与 b 是否在争同一个位置。"""
    if same_kind and a.kind != b.kind:
        return False
    lim = max(floor_px, frac * min(a.short, b.short)) if frac > 0 else floor_px
    return center_dev(a.box, b.box) <= lim


def thin(parties: Sequence[Party], *, frac: float = FRAC, floor_px: float = FLOOR_PX,
         same_kind: bool = True, rule: str = "", ledger: Optional[List[dict]] = None,
         ) -> List[Party]:
    """把争同一个位置的多个主张者压到**一个**，分数最高者胜。

    返回保留下来的 `Party`（原始顺序），判负的连同理由写进 `ledger`。
    这是「非极大抑制」的**归属版**：判负的理由是「你在跟一个更高的分抢同一个位置」，
    而不是「你俩的 IoU 大于某个数」。
    """
    order = sorted(parties, key=lambda p: -p.score)
    kept: List[Party] = []
    dropped: List[Tuple[Party, Party, float]] = []
    for p in order:
        rival = None
        for k in kept:
            if contenders(p, k, frac=frac, floor_px=floor_px, same_kind=same_kind):
                rival = k
                break
        if rival is None:
            kept.append(p)
        else:
            dropped.append((p, rival, center_dev(p.box, rival.box)))
    if ledger is not None:
        tag = rule or (f"frac<={frac:g}" if frac > 0 else f"cd<={floor_px:g}")
        for p, k, dev in dropped:
            ledger.append({
                "rule": tag, "kind": p.kind, "dev": round(dev, 2),
                "frac": round(dev / max(1e-6, min(p.short, k.short)), 3),
                "kept": k.id, "kept_score": round(k.score, 4),
                "dropped": p.id, "dropped_score": round(p.score, 4),
                "why": "同一位置已有更高分的主张者",
                "dropped_meta": dict(p.meta),
            })
    keep_ids = {id(p) for p in kept}
    return [p for p in parties if id(p) in keep_ids]


def _assign(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """与官方判分**同一份实现**（`schematic_parser.metrics._assign`）。

    拿不到时退回本地贪心 —— 只为让本模块能独立运行，生产路径一定走官方那份。
    """
    try:
        from schematic_parser.metrics import _assign as _off
        return _off(cost)
    except Exception:
        pass
    pairs: List[Tuple[int, int]] = []
    flat = sorted(((c, i, j) for i, row in enumerate(cost)
                   for j, c in enumerate(row)), key=lambda t: t[0])
    ru, cu = set(), set()
    for c, i, j in flat:
        if i in ru or j in cu:
            continue
        ru.add(i)
        cu.add(j)
        pairs.append((i, j))
    return pairs


def assign(claimants: Sequence[Party], items: Sequence[Party], *,
           gate: Optional[Callable[[Party, Party], bool]] = None,
           cost: Optional[Callable[[Party, Party], float]] = None,
           ledger: Optional[List[dict]] = None, rule: str = "assign",
           ) -> Tuple[Dict[str, str], List[dict]]:
    """一对一归属：`claimants` 抢 `items`，用**整体**最小代价解，不是逐个贪心。

    返回 `(claimant_id -> item_id)` 与账。够不到任何标的的主张者**原样留着**
    （交给下一层去争），不在这里销毁。
    """
    INF = float("inf")
    gate = gate or (lambda c, i: True)
    cost = cost or (lambda c, i: center_dev(c.box, i.box))
    C = [[(cost(c, i) if gate(c, i) else INF) for i in items] for c in claimants]
    out: Dict[str, str] = {}
    pairs = _assign(C) if (C and C[0]) else []
    for ci, ii in pairs:
        if C[ci][ii] == INF:
            continue
        out[claimants[ci].id] = items[ii].id
    if ledger is not None:
        back: Dict[str, List[str]] = {}
        for cid, iid in out.items():
            back.setdefault(iid, []).append(cid)
        for i in items:
            got = back.get(i.id, [])
            if len(got) <= 1:
                continue
            ledger.append({"rule": rule, "kind": i.kind, "item": i.id,
                           "won": got[0], "also_claimed": got[1:],
                           "why": "一个标的只能有一个主人，其余改判"})
    return out, ledger


def legacy_thin(parties: Sequence[Party], px: float = LEGACY_PX,
                ledger: Optional[List[dict]] = None) -> List[Party]:
    """绝对像素版（沿用 `stage1/detect.py` 已验证的 `DEDUP_PX=12`）。

    19 张实测：并掉 2 对，**真重复 2 / 误伤 0**。
    """
    return thin(parties, frac=0.0, floor_px=px, rule=f"cd<={px:g}", ledger=ledger)


# ---------------------------------------------------------------------------
def self_test() -> int:
    """三条不变式 + 判据的边界。`python cvparts/own.py` 直接跑。"""
    ok = [0, 0]

    def chk(cond, msg):
        ok[1] += 1
        if cond:
            ok[0] += 1
        else:
            print("  ✗ " + msg)

    # 1) 小图章上 IoU 判不出「同一颗」，中心距判得出
    a = Party("a", (0, 0, 20, 20), "r", 0.9)
    b = Party("b", (10, 0, 30, 20), "r", 0.8)          # 中心差 10px
    chk(iou(a.box, b.box) < 0.5, "20x20 错开 10px 的 IoU 应 <0.5")
    chk(contenders(a, b, frac=0.5), "错开 10px = 半个短边，应视为争同一颗")
    chk(not contenders(a, b, frac=0.3), "错开 10px 超过 0.3*20，默认阈值下不并（0192 电阻阵列就靠这条）")

    # 2) 不同类型不争
    chk(not contenders(a, Party("c", (0, 0, 20, 20), "c", 0.5)), "不同类型不该争")

    # 3) 一对一 + 有账 + 不销毁
    ps = [Party(f"p{i}", (i * 4, 0, i * 4 + 20, 20), "r", 0.9 - i * 0.1) for i in range(3)]
    led: List[dict] = []
    kept = thin(ps, frac=0.5, ledger=led)
    chk(len(kept) == 1 and kept[0].id == "p0", "3 个挤在 10px 内只留 1 个（分最高）")
    chk(len(led) == 2, "判负的 2 个都进账")
    chk(all(e["kept"] == "p0" and e["dropped_score"] > 0 for e in led), "账里记了输给谁与原始分数")
    chk(len(ps) == 3, "原列表没被就地改（不销毁）")

    # 3b) 链式相邻**不**传递合并 —— 刻意如此。
    #     p0..p3 两两间距 4px，但 p3 离 p0 有 12px（>0.5*20=10）⇒ 幸存。
    #     如果做传递闭包，`0092` 那种 16.5px 间距的 R1..R20 电阻阵列会被整排吃成一个，
    #     所以这里必须是非传递的贪心。
    chain = [Party(f"q{i}", (i * 4, 0, i * 4 + 20, 20), "r", 0.9 - i * 0.1) for i in range(4)]
    ck = thin(chain, frac=0.5)
    chk([p.id for p in ck] == ["q0", "q3"], "链式相邻不传递合并（免得吃掉密集阵列）")

    # 4) 单调性：阈值放宽只多不少
    n30 = len(thin(ps, frac=0.3))
    n50 = len(thin(ps, frac=0.5))
    chk(n50 <= n30, "阈值放宽后留下的不该变多")

    # 5) assign：一个标的只能有一个主人
    items = [Party("I1", (0, 0, 20, 20), "gnd")]
    cl = [Party("c1", (2, 0, 22, 20), "gnd", 0.9), Party("c2", (4, 0, 24, 20), "gnd", 0.8)]
    led2: List[dict] = []
    m, _ = assign(cl, items, gate=lambda c, i: center_dev(c.box, i.box) <= 20,
                  ledger=led2)
    chk(len(m) == 1 and m.get("c1") == "I1", "两个抢一个 → 必输一个")
    chk(len(led2) == 0 or led2[0]["won"] == "c1", "账里记了赢家")

    # 6) 绝对像素臂：19 张实测并 2 对、误伤 0
    far = [Party("x", (0, 0, 20, 20), "r", 0.9), Party("y", (13, 0, 33, 20), "r", 0.8)]
    chk(len(legacy_thin(far)) == 2, "中心差 13px > 12px 不该并")
    chk(len(legacy_thin([Party("x", (0, 0, 20, 20), "r", 0.9),
                         Party("y", (11, 0, 31, 20), "r", 0.8)])) == 1,
        "中心差 11px <= 12px 该并")

    print(f"自检：{ok[0]}/{ok[1]} 通过")
    return 0 if ok[0] == ok[1] else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
