"""统一交叉校验层（「大一统」）：Claim / Verdict / Check / Registry / Agenda。

**为什么需要这一层**（`docs/reports/统一校验与裁决层_侦察报告.md`）：
全仓 45 处交叉校验各自为政 —— 候选集至少有 5 种并行表示（`desig_all`、`Slot/Alt`、
`ClaimPool/Region`、`fusion.axis.Item/Axis`、`structmatch.rebind` 的 `pool` 形参），
「本来会被丢掉的东西」有 **6 套互不兼容的留痕载体**（`LEDGER.demote/record`、
`LAST_CONFORM`/`LAST_CONFORM_EXTRA`、`LAST_DEMOTED`/`LAST_RESERVE`、`designator_dropped`、
`own.thin(ledger=)`、`plausible/priors` 的 `sus` 清单），裁决形状有 3 种
（`Demotion` / `Event` / 裸 dict）。结果是：**同一个「否决」动作，在不同阶段的
可追溯性、可复算性、可回滚性完全不同**，复核者要按 6 套格式各读一遍。

这一层**只加不改**（`docs/工程哲学.md` 的四条原则）：
* P1 只加不改 —— 本模块不 import 任何 stage 模块的私有状态，不修改现有类。
* P2 判负必留痕 —— 任何「本该被丢掉」的东西都落成 `Verdict`，四档降权、**永不取 0**。
* P3 逐位不变优先 —— 本层默认**只读**；`CV_UNIFIED=0`（默认）时管线一行都不走这里。
* P4 不引第三方 —— 只用标准库。

**高效中间过程数据结构**（用户 m50175「还要提供各种各样的支持，比如说对应的高效中间
过程的数据结构」）：`Registry` 一次性建好三种索引，供所有校验共用，避免每个校验各自
O(n·m) 重扫：
* `by_slot`  —— 槽位 → 候选（按分降序，惰性重排）
* `by_value` —— 取值 → 声明它的槽位（位号唯一性、「撞号」冲突一次算清）
* `GridIndex` —— 槽位取景框的空间哈希（`near(box, r)` 从 O(n) 降到只看邻近格）
* `AlongIndex` —— 「元件 + 边」→ 沿边坐标有序数组（单调性/顺序类校验用 `bisect`）

CLI（只读诊断）::

    python -X utf8 -m cvparts.unified --dir work/_fast_out_anbdef10 --case 5
    python -X utf8 -m cvparts.unified            # 自检
"""

from __future__ import annotations

import bisect
import heapq
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .hypo import (DEMOTE_HARD, DEMOTE_OVERWRITE, DEMOTE_SOFT, DEMOTE_TRUNC,
                   LEDGER, Event, Ledger)

# 本层自身不改变管线行为：只有显式打开（或在诊断里手动调用）才会跑校验。
UNIFIED = os.environ.get("CV_UNIFIED", "0")
GRID_CELL = float(os.environ.get("CV_GRID_CELL", "32"))
AGENDA_MAX = int(os.environ.get("CV_AGENDA_MAX", "400"))
# 位号取景框与「它自己那颗元件」之间允许的最大距离（几何判据，见 check_desig_near_comp）
DESIG_NEAR = float(os.environ.get("CV_DESIG_NEAR", "60"))

# 阶段标记（写进 `Verdict.at`，与既有 `Demotion.at` 的习惯一致）
S1, S2, S3 = "stage1", "stage2", "stage3"

# 槽位种类（`slot_id` 的前缀）。新增种类只需在这里登记，索引与议程自动生效。
K_DESIG = "desig"        # 位号（元件键）
K_PIN = "pin"            # 引脚**几何点**（stage1 的脚点，只有坐标还没读值）
K_PIN_NUM = "pin_number"  # 引脚号
K_PIN_NAME = "pin_name"   # 引脚名
K_NET = "net"             # 网络成员集
K_LINE = "net_line"       # 导线通道
K_TYPE = "type"           # 元件类型
K_PART = "partname"       # 型号/partname 印字
K_COMP = "comp"           # 元件本体（框）
K_ANY = "any"             # 只对某一种成立的校验用不上它

# ★ **唯一性域**：只有这些种类的取值在全局必须唯一。位号（元件键）是经典的唯一性
#   约束；而**脚号/脚名只在同一颗元件内唯一**（实测 case 5 的 5 颗 IC 各自都有 `pin_2`，
#   裸 `conflicts()` 会报 14 条撞号、14 条全是这个，纯噪声），网络成员集更是天然重复。
#   把所有 `by_value` 撞号一律当成问题，只会训练人忽略这个字段。
UNIQUE_KINDS: Tuple[str, ...] = (K_DESIG,)


# --------------------------------------------------------------------------- 类型
@dataclass
class Claim:
    """一条候选证据：**某个槽位可以取某个值**，附分与来源。

    这是 5 种并行候选表示的**公共投影**：`desig_all[cid][i]`、`hypo.Alt`、
    `fusion.claims` 里的池项、`axis.Item`、`structmatch` 的 `pool` 元素都能投影成它，
    但**反过来不要求**（本层不接管它们的生命周期）。
    """

    slot: str
    value: Any = None
    score: float = 0.0
    source: str = ""            # 谁报的（"stage2:desig"、"vlm"、"pincol"…）
    by: str = ""                # 哪个阶段的产物
    evidence: dict = field(default_factory=dict)
    region: Optional[Tuple[float, float, float, float]] = None
    attrs: dict = field(default_factory=dict)   # edge/along/raw_text…（校验专用）
    at: str = ""

    @property
    def kind(self) -> str:
        return self.slot.split(":", 1)[0]

    def key(self) -> str:
        try:
            return json.dumps(self.value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(self.value)


@dataclass
class Verdict:
    """一次校验的结论：**本来会被丢掉的东西**改成什么。

    `kind` 就是 `cvparts.hypo` 的四档降权 + `keep`：

    ===========  ======  ===============================================
    kind         分值    含义
    ===========  ======  ===============================================
    ``hard``     0.05    判据被强证据否掉（原来是 0 = 删除）
    ``soft``     0.15    与主流不一致（弱判据）
    ``trunc``    0.02    被截断/挤出（超上限、超预算）
    ``overwrite``0.05    被别的证据改写
    ``keep``     —       校验通过，不改分（也要留痕：它解释「为什么没被丢」）
    ===========  ======  ===============================================
    """

    slot: str
    kind: str = "soft"
    before: float = 0.0         # 判定**之前**的分（house style 的 score_before）
    after: Any = None           # 这个槽位最终持有的值/绑定（不是分）
    rule: str = ""
    by: str = ""
    at: str = ""
    detail: dict = field(default_factory=dict)

    _DEMOTE = {"hard": DEMOTE_HARD, "soft": DEMOTE_SOFT, "trunc": DEMOTE_TRUNC,
               "overwrite": DEMOTE_OVERWRITE}

    @property
    def score_before(self) -> float:
        try:
            return float(self.before or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def score_after(self) -> float:
        """降权后的分（`keep` 保持原值）。"""
        return self.score_before if self.kind == "keep" else self._DEMOTE.get(self.kind, DEMOTE_SOFT)

    @property
    def value(self) -> Any:
        return self.after

    @property
    def lost(self) -> float:
        return round(self.score_before - self.score_after, 6)

    def as_event(self) -> Event:
        return Event(kind=f"verdict:{self.kind}", where=self.slot,
                     before=[] if self.after is None else [self.after],
                     after=self.after, reason=self.rule, by=self.by, at=self.at)

    def as_demotion(self):
        from .hypo import Demotion
        return Demotion(where=self.slot, value=self.after,
                        score_before=round(self.score_before, 4),
                        score_after=self.score_after,
                        reason=self.rule, by=self.by, at=self.at)


@dataclass
class Check:
    """一条交叉校验：**输入是注册表 + 一个槽位，输出是若干 `Verdict`**。

    这是「把现有各路校验接口泛化」的落点 —— 现有的三种校验形态都能包进来：

    * 已经返回 `Verdict` 的：直接 `Check(name, kinds, run=fn)`
    * 返回裸 dict / (kind, reason) 的：`Check.adapt(fn)`
    * 只返回布尔（「不合格」）的：`Check.predicate(fn, fail_kind="soft", rule=...)`
    """

    name: str
    run: Callable[["Registry", str], Iterable[Verdict]]
    kinds: Tuple[str, ...] = (K_ANY,)
    by: str = ""
    at: str = ""
    enabled: bool = True
    docs: str = ""

    def applies(self, slot: str) -> bool:
        k = slot.split(":", 1)[0]
        return K_ANY in self.kinds or k in self.kinds

    # ---- 三种历史形态的适配器 ----
    @staticmethod
    def adapt(fn: Callable[..., Any], name: str = "", kinds: Tuple[str, ...] = (K_ANY,),
              by: str = "", at: str = "", docs: str = "") -> "Check":
        """包一个「返回裸 dict 或 (kind, reason, after)」的旧校验。"""
        def run(reg: "Registry", slot: str) -> Iterable[Verdict]:
            out = fn(reg, slot)
            if out is None:
                return []
            if isinstance(out, Verdict):
                return [out]
            items = out if isinstance(out, (list, tuple)) else [out]
            vs: List[Verdict] = []
            for it in items:
                if isinstance(it, Verdict):
                    vs.append(it)
                    continue
                if isinstance(it, dict):
                    vs.append(Verdict(slot=slot, by=by, at=at, **it))
                elif isinstance(it, (list, tuple)) and it:
                    k = str(it[0])
                    r = str(it[1]) if len(it) > 1 else ""
                    a = it[2] if len(it) > 2 else None
                    vs.append(Verdict(slot=slot, kind=k, rule=r, after=a, by=by, at=at))
            return vs
        return Check(name=name or getattr(fn, "__name__", "check"), run=run,
                     kinds=kinds, by=by, at=at, docs=docs)

    @staticmethod
    def predicate(fn: Callable[[Any], bool], name: str = "", kinds: Tuple[str, ...] = (K_ANY,),
                  fail_kind: str = "soft", rule: str = "", by: str = "",
                  at: str = "", docs: str = "") -> "Check":
        """包一个「返回 True 表示合格」的旧判据：不合格 ⇒ 一条 `fail_kind` 降权。"""
        def run(reg: "Registry", slot: str) -> Iterable[Verdict]:
            sv = reg.slot(slot)
            best = sv.best if sv else None
            ok = bool(fn(best.value if best else None, sv)) if _takes_two(fn) else \
                bool(fn(best.value if best else None))
            if ok:
                return [Verdict(slot=slot, kind="keep", before=(best.score if best else 0.0),
                                after=(best.value if best else None), rule=rule or "合格",
                                by=by, at=at)]
            return [Verdict(slot=slot, kind=fail_kind, before=(best.score if best else 0.0),
                            after=(best.value if best else None), rule=rule or "不合格",
                            by=by, at=at)]
        return Check(name=name or getattr(fn, "__name__", "check"), run=run,
                     kinds=kinds, by=by, at=at, docs=docs)

    # ---- 第四种形态：整池审计（返回可疑项清单） ----
    @staticmethod
    def from_issues(fn: Callable[["Registry"], Iterable[dict]], name: str = "",
                    kinds: Tuple[str, ...] = (K_ANY,), fail_kind: str = "soft",
                    map_slot: Optional[Callable[[str], Optional[str]]] = None,
                    by: str = "", at: str = "", docs: str = "") -> "Check":
        """包一个「跑一遍整池、返回可疑项清单」的旧审计。

        `fusion/plausible.check(pool)` 与 `fusion/priors.Priors.check(pool)` 都是这个形态：
        每项 `{kind, severity, why, slots, nets, hint, need_llm}`，`slots` = **可去重扫的槽位**。
        这里把 `why` 记成判定理由、`severity` 折算档位（`>= 0.7` ⇒ `hard`），
        每条 issue 里的每个 `slots` 各收一条判定；整池审计**每个注册表只跑一次**
        （结果按槽位缓存），而不是每个槽位跑一遍 —— 这正是「高效中间过程」的意思。
        `map_slot`：旧审计用它自己的槽位命名（如 `comp.c1.key`），给个翻译函数即可接上。
        """
        cache: Dict[int, Dict[str, List[Verdict]]] = {}

        def run(reg: "Registry", slot: str) -> Iterable[Verdict]:
            if id(reg) not in cache:
                by_slot: Dict[str, List[Verdict]] = {}
                for it in (fn(reg) or []):
                    if not isinstance(it, dict):
                        continue
                    why = str(it.get("why") or it.get("rule") or "")
                    kind = _sev_kind(it, fail_kind)
                    for raw in (it.get("slots") or []):
                        s = map_slot(str(raw)) if map_slot else str(raw)
                        if not s:
                            continue
                        sv = reg.slot(s)
                        by_slot.setdefault(s, []).append(Verdict(
                            slot=s, kind=kind,
                            before=(sv.best.score if sv and sv.best else 0.0),
                            after=(sv.best.value if sv and sv.best else None),
                            rule=why, by=by, at=at, detail=it))
                cache[id(reg)] = by_slot
            return cache[id(reg)].get(slot, [])

        return Check(name=name or getattr(fn, "__name__", "issues"), run=run,
                     kinds=kinds, by=by, at=at, docs=docs)


def _takes_two(fn: Callable) -> bool:
    import inspect
    try:
        return len([p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2
    except (TypeError, ValueError):
        return False


def _sev_kind(issue: dict, default: str = "soft") -> str:
    """旧审计给的严重度 → 统一档位：显式 `kind` 优先，否则 `severity >= 0.7` 记 `hard`。"""
    k = str(issue.get("verdict_kind") or issue.get("kind") or "")
    if k in ("hard", "soft", "trunc", "overwrite", "keep"):
        return k
    try:
        return "hard" if float(issue.get("severity") or 0.0) >= 0.7 else default
    except (TypeError, ValueError):
        return default


@dataclass
class SlotView:
    """一个槽位的统一视图（惰性重排，避免每加一条候选就 sort 一次）。"""

    slot: str
    claims: List[Claim] = field(default_factory=list)
    _sorted: bool = True
    decided: Any = None
    decided_by: Optional[str] = None
    region: Optional[Tuple[float, float, float, float]] = None
    attrs: dict = field(default_factory=dict)

    def add(self, c: Claim) -> None:
        """同一个值被多个来源支持 ⇒ **合并**（取最高分、证据并起来），与 `hypo.Slot.add` 同义。"""
        for ex in self.claims:
            if ex.key() == c.key():
                if float(c.score) > float(ex.score):
                    ex.score = float(c.score)
                    self._sorted = False
                ex.evidence.setdefault("also_from", []).append(
                    {"source": c.source, "by": c.by, "score": float(c.score)})
                if ex.region is None and c.region is not None:
                    ex.region = c.region
                for k, v in (c.attrs or {}).items():
                    ex.attrs.setdefault(k, v)
                return
        self.claims.append(c)
        self._sorted = False
        if c.region is not None and self.region is None:
            self.region = c.region
        for k, v in (c.attrs or {}).items():
            self.attrs.setdefault(k, v)

    def _ensure(self):
        if not self._sorted:
            self.claims.sort(key=lambda c: -float(c.score))
            self._sorted = True

    @property
    def best(self) -> Optional[Claim]:
        self._ensure()
        return self.claims[0] if self.claims else None

    @property
    def second(self) -> Optional[Claim]:
        self._ensure()
        return self.claims[1] if len(self.claims) > 1 else None

    @property
    def margin(self) -> float:
        b, s = self.best, self.second
        return (float(b.score) - float(s.score)) if (b and s) else float("inf")

    @property
    def kind(self) -> str:
        return self.slot.split(":", 1)[0]

    def ambiguous(self, margin: float = 0.15) -> bool:
        return self.second is not None and self.margin < margin


# --------------------------------------------------------------------------- 索引
class GridIndex:
    """槽位取景框的空间哈希：`near(box, r)` 只扫邻近格。

    没有它时每条几何校验都要 `for sv in reg.slots()` 全扫一遍 —— 官方图 ~150 个槽位、
    派生图 ~400 个，45 条校验各扫一遍就是 45·n。这里是 O(1) 摊销。
    """

    def __init__(self, cell: float = GRID_CELL):
        self.cell = max(1.0, float(cell))
        self.cells: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        self.boxes: Dict[str, Tuple[float, float, float, float]] = {}
        self.n_query = 0

    def add(self, slot: str, box: Sequence[float]) -> None:
        b = _box4(box)
        if b is None:
            return
        self.boxes[slot] = b
        for c in self._cells(b):
            self.cells[c].append(slot)

    def _cells(self, b) -> List[Tuple[int, int]]:
        c = self.cell
        x0, y0 = int(math.floor(b[0] / c)), int(math.floor(b[1] / c))
        x1, y1 = int(math.floor(b[2] / c)), int(math.floor(b[3] / c))
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 4096:      # 巨大的框：别把整张图枚举一遍
            return [(x0, y0)]
        return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def near(self, box: Sequence[float], r: float = 0.0,
             exclude: Optional[str] = None) -> List[str]:
        """返回与 `box` 外扩 `r` 后**真的相交/相邻**的槽位（按距离升序）。"""
        b = _box4(box)
        if b is None:
            return []
        self.n_query += 1
        grown = (b[0] - r, b[1] - r, b[2] + r, b[3] + r)
        seen, hits = set(), []
        for c in self._cells(grown):
            for s in self.cells.get(c, ()):
                if s == exclude or s in seen:
                    continue
                seen.add(s)
                d = box_dist(grown, self.boxes[s])
                if d <= 0.0:
                    hits.append((box_dist(b, self.boxes[s]), s))
        hits.sort()
        return [s for _d, s in hits]


class AlongIndex:
    """「元件 + 边」→ 沿边坐标的有序数组（单调/顺序类校验用 `bisect` 定位邻居）。"""

    def __init__(self):
        self.rows: Dict[Tuple[str, str], List[Tuple[float, str]]] = defaultdict(list)

    def add(self, slot: str, attrs: dict) -> None:
        key = (str(attrs.get("cid", "")), str(attrs.get("edge", "")))
        along = attrs.get("along")
        if along is None or not key[1]:
            return
        self.rows[key].append((float(along), slot))

    def build(self) -> None:
        for k in self.rows:
            self.rows[k].sort()

    def rank(self, slot: str, attrs: dict) -> Optional[int]:
        """该槽位在它那条边上的序（0 起）；没建索引返回 None。"""
        key = (str(attrs.get("cid", "")), str(attrs.get("edge", "")))
        row = self.rows.get(key)
        if not row:
            return None
        langs = [a for a, _s in row]
        along = attrs.get("along")
        if along is None:
            return None
        i = bisect.bisect_left(langs, float(along))
        return i

    def neighbors(self, slot: str, attrs: dict, k: int = 1) -> List[str]:
        """沿边最近的 `k` 个**别的**槽位（按 |沿边差| 升序，平票时取坐标小的）。"""
        row = self.rows.get((str(attrs.get("cid", "")), str(attrs.get("edge", ""))))
        if not row:
            return []
        langs = [a for a, _s in row]
        along = float(attrs.get("along", 0.0))
        i = bisect.bisect_left(langs, along)
        cand = []
        for j in range(max(0, i - k), min(len(row), i + k + 1)):
            if row[j][1] != slot:
                cand.append((abs(row[j][0] - along), row[j][0], row[j][1]))
        cand.sort()
        return [s for _d, _a, s in cand[:k]]


# --------------------------------------------------------------------------- 注册表
class Registry:
    """一张图上的**全部候选证据** + 三种索引 + 校验登记（本层的中心对象）。"""

    def __init__(self, case: str = "", ledger: Optional[Ledger] = None):
        self.case = str(case)
        self.ledger = ledger or LEDGER
        self.by_slot: Dict[str, SlotView] = {}
        self.by_value: Dict[str, List[str]] = defaultdict(list)
        self.kinds: Dict[str, List[str]] = defaultdict(list)
        self.claims: List[Claim] = []
        self.checks: Dict[str, Check] = {}
        self.verdicts: List[Verdict] = []
        self.grid: Optional[GridIndex] = None
        self.along: Optional[AlongIndex] = None
        self._grid_dirty = True
        self._along_dirty = True
        self.n_added = 0

    # ---- 写 ----
    def add(self, c: Claim) -> Claim:
        sv = self.by_slot.get(c.slot)
        if sv is None:
            sv = self.by_slot[c.slot] = SlotView(slot=c.slot)
            self.kinds[c.kind].append(c.slot)
        sv.add(c)
        self.claims.append(c)
        if c.value is not None:            # 空值不参与「撞值」判定（脚点/几何槽没有值）
            self.by_value[c.key()].append(c.slot)
        self._grid_dirty = self._along_dirty = True
        self.n_added += 1
        return c

    def add_many(self, cs: Iterable[Claim]) -> int:
        n = 0
        for c in cs:
            self.add(c)
            n += 1
        return n

    def decide(self, slot: str, value: Any, by: str = "") -> None:
        sv = self.by_slot.get(slot)
        if sv is not None:
            sv.decided, sv.decided_by = value, by or "registry"

    # ---- 读 ----
    def slot(self, slot_id: str) -> Optional[SlotView]:
        return self.by_slot.get(slot_id)

    def slots(self, kind: Optional[str] = None) -> List[str]:
        if kind is None:
            return list(self.by_slot)
        if kind == K_ANY:
            return list(self.by_slot)
        return list(self.kinds.get(kind, ()))

    def value_slots(self, value: Any) -> List[str]:
        try:
            k = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            k = str(value)
        return list(dict.fromkeys(self.by_value.get(k, ())))

    def conflicts(self, kinds: Optional[Sequence[str]] = None) -> List[Tuple[Any, List[str]]]:
        """「同一个取值被两个槽位同时声称」（位号唯一性这类硬约束）一次算清。"""
        out = []
        for k, slots in self.by_value.items():
            u = list(dict.fromkeys(slots))
            if len(u) < 2:
                continue
            if kinds is not None and not any(s.split(":", 1)[0] in kinds for s in u):
                continue
            out.append((k, u))
        return out

    def index(self) -> GridIndex:
        if self.grid is None:
            self.grid = GridIndex()
        if self._grid_dirty:
            self.grid = GridIndex()
            for s, sv in self.by_slot.items():
                if sv.region is not None:
                    self.grid.add(s, sv.region)
            self._grid_dirty = False
        return self.grid

    def along_index(self) -> AlongIndex:
        if self.along is None or self._along_dirty:
            self.along = AlongIndex()
            for s, sv in self.by_slot.items():
                self.along.add(s, sv.attrs)
            self.along.build()
            self._along_dirty = False
        return self.along

    def near(self, box: Sequence[float], r: float = 0.0,
             exclude: Optional[str] = None, kinds: Optional[Sequence[str]] = None) -> List[str]:
        hits = self.index().near(box, r, exclude)
        if kinds is None:
            return hits
        return [s for s in hits if s.split(":", 1)[0] in kinds]

    # ---- 校验 ----
    def register(self, *checks: Check) -> None:
        for c in checks:
            self.checks[c.name] = c

    def run(self, only: Optional[Sequence[str]] = None,
            ledger: Optional[Ledger] = None) -> List[Verdict]:
        """跑所有启用的校验，**逐条留痕**（`Verdict` → 账本）。"""
        led = ledger or self.ledger
        out: List[Verdict] = []
        for name, ck in self.checks.items():
            if not ck.enabled or (only is not None and name not in only):
                continue
            for slot in self.slots():
                if not ck.applies(slot):
                    continue
                for v in (ck.run(self, slot) or ()):
                    if not isinstance(v, Verdict):
                        continue
                    if not v.by:
                        v.by = ck.by or ck.name
                    if not v.at:
                        v.at = ck.at
                    out.append(v)
                    self.record_verdict(v, led)
        self.verdicts.extend(out)
        return out

    def record_verdict(self, v: Verdict, ledger: Optional[Ledger] = None) -> None:
        """`Verdict` → 既有账本（`Demotion` + `Event`），让新旧留痕汇到一处。"""
        led = ledger or self.ledger
        if v.kind != "keep":
            led.demote(v.slot, v.after, v.score_before, v.score_after,
                       reason=v.rule, by=v.by, at=v.at or "unified")
        led.record(f"verdict:{v.kind}", v.slot,
                   before=[] if v.after is None else [v.after],
                   after=v.after, reason=v.rule, by=v.by, at=v.at or "unified")

    # ---- 汇总 ----
    def summary(self) -> dict:
        amb = [s for s, sv in self.by_slot.items() if sv.ambiguous()]
        return {"case": self.case, "n_claims": self.n_added, "n_slots": len(self.by_slot),
                "n_vals": len(self.by_value),
                "by_kind": {k: len(v) for k, v in sorted(self.kinds.items())},
                "n_ambiguous": len(amb), "n_conflicts": len(self.conflicts()),
                "n_uniq_conflicts": len(self.conflicts(UNIQUE_KINDS)),
                "n_checks": len(self.checks), "n_verdicts": len(self.verdicts),
                "n_grid_query": (self.grid.n_query if self.grid else 0)}

    def to_json(self, top: int = 8) -> dict:
        d = self.summary()
        d["slots"] = [{"slot": s, "best": (sv.best.value if sv.best else None),
                       "score": round(float(sv.best.score), 4) if sv.best else 0.0,
                       "margin": (round(sv.margin, 4) if sv.second else None),
                       "n": len(sv.claims)}
                      for s, sv in sorted(self.by_slot.items(),
                                          key=lambda kv: -(kv[1].best.score if kv[1].best else 0))[:top]]
        # ★ 只报**唯一性域**（位号）的撞号：脚号/脚名在语义上只要求「同一颗元件内不重复」
        #   （实测 case 5 的 14 条裸撞号里 14 条都是不同 IC 各自有 pin_2 —— 正常），
        #   把它们当撞号报出来只会训练人忽略这个字段。
        d["conflicts"] = [{"value": k, "slots": v} for k, v in self.conflicts(UNIQUE_KINDS)[:top]]
        return d


# --------------------------------------------------------------------------- 议程
class Agenda:
    """待裁决队列：**只挑真正需要算力的槽位**（`工程哲学.md:94` 第 ③ 条）。

    优先级 = 候选分差（`margin`）：分差越小越可能翻盘。已经裁决过、或只有一个候选的
    槽位**不进队列**（它们不需要复核）。`Registry.index()` 让「哪些槽位彼此有冲突」
    也变成一次查表，而不是每次重扫。
    """

    def __init__(self, reg: Registry, margin: float = 0.15, cap: int = AGENDA_MAX):
        self.reg = reg
        self.margin = float(margin)
        self.cap = int(cap)
        self.heap: List[Tuple[float, int, str]] = []
        self._seq = 0
        self.done: List[str] = []

    def push(self, slot: str, prio: float = 0.0) -> None:
        if len(self.heap) >= self.cap:
            return
        self._seq += 1
        heapq.heappush(self.heap, (-float(prio), self._seq, slot))

    def pop(self) -> Optional[str]:
        if not self.heap:
            return None
        s = heapq.heappop(self.heap)[2]
        self.done.append(s)
        return s

    def pending(self) -> List[str]:
        return [s for _p, _q, s in sorted(self.heap)]

    def __len__(self) -> int:
        return len(self.heap)

    @classmethod
    def from_registry(cls, reg: Registry, margin: float = 0.15,
                      cap: int = AGENDA_MAX) -> "Agenda":
        a = cls(reg, margin=margin, cap=cap)
        for s, sv in reg.by_slot.items():
            if sv.decided is not None or sv.second is None:
                continue                      # 已裁决 / 只有一个候选 ⇒ 不需要复核
            a.push(s, sv.margin if math.isfinite(sv.margin) else 1e9)
        return a

    @classmethod
    def from_pool(cls, reg: Registry, pool: Any, margin: float = 0.15,
                  cap: int = AGENDA_MAX) -> "Agenda":
        """把任一历史候选池接进来（都是**只读**投影）。

        认识：`ClaimPool`（有 `.hy`）、`CaseHypotheses`（有 `.slots`）、
        `{slot: Slot}`、`{cid: [候选 dict]}`（`desig_all` 的形状）。
        """
        reg.add_many(project_pool(pool))
        return cls.from_registry(reg, margin=margin, cap=cap)


# --------------------------------------------------------------------------- 投影（只读）
def _box4(b) -> Optional[Tuple[float, float, float, float]]:
    if not b or len(b) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except (TypeError, ValueError):
        return None
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def box_dist(a, b) -> float:
    """两个轴对齐框的距离（相交/相切 = 0）。"""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def point_box_dist(p, b) -> float:
    dx = max(b[0] - p[0], p[0] - b[2], 0.0)
    dy = max(b[1] - p[1], p[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def project_pool(pool: Any, by: str = "", prefix: str = "") -> List[Claim]:
    """把历史候选池投影成 `Claim` 列表（`desig_all` / `Slot` / `CaseHypotheses` / `ClaimPool`）。"""
    out: List[Claim] = []
    if pool is None:
        return out
    slots = None
    if hasattr(pool, "hy"):                       # ClaimPool（fusion.claims）
        slots = getattr(pool.hy, "slots", None)
        by = by or "fusion:claims"
        regions = getattr(pool, "regions", {}) or {}
    elif hasattr(pool, "slots"):                  # CaseHypotheses / {slot: Slot}
        slots = pool.slots
        regions = {}
        by = by or "cvparts:hypo"
    elif isinstance(pool, dict):                  # {slot|"cid": [...]}
        for k, v in pool.items():
            if isinstance(v, dict):               # {slot: Slot}
                sv = v
                out.extend(_claims_of_slot(prefix + str(k), sv, None, by))
            else:                                 # desig_all 形状：cid -> [候选 dict]
                out.extend(_claims_of_desig(prefix + str(k), v, by))
        return out
    else:
        return out
    if isinstance(slots, dict):
        for sid, sv in slots.items():
            reg = regions.get(sid) if isinstance(regions, dict) else None
            out.extend(_claims_of_slot(prefix + str(sid), sv, reg, by))
    return out


def _claims_of_slot(slot: str, sv: Any, region: Any, by: str) -> List[Claim]:
    out = []
    box = _box4(getattr(region, "box", None)) if region is not None else None
    for a in (getattr(sv, "alts", None) or []):
        out.append(Claim(slot=str(slot), value=getattr(a, "value", None),
                         score=float(getattr(a, "score", 0.0) or 0.0),
                         source=str(getattr(a, "source", "") or ""), by=by,
                         evidence=dict(getattr(a, "evidence", {}) or {}),
                         region=box, attrs=dict(getattr(sv, "context", {}) or {})))
    return out


def _claims_of_desig(cid: str, alts: Any, by: str) -> List[Claim]:
    out = []
    for a in (alts or []):
        if not isinstance(a, dict):
            continue
        out.append(Claim(slot=f"{K_DESIG}:{cid}", value=a.get("canon", a.get("text")),
                         score=float(a.get("score", 0.0) or 0.0),
                         source=str(a.get("fit", "") or ""), by=by or S2,
                         evidence={k: a.get(k) for k in ("text", "conf", "d", "box_i", "type_fix")},
                         region=_box4(a.get("box")),
                         attrs={"raw_text": a.get("text")}))
    return out


def project_stage1(doc: dict, by: str = S1) -> List[Claim]:
    """stage1 产物 → `Claim`：元件框、引脚几何点 / 沿边列（`pincol`）、网络成员集。

    注意 stage1 的 `pincol` **只有几何**（`side/idx/x/y/axis/pitch`），号码与名字是
    stage2 才读出来的 ⇒ 这里只登记 `K_PIN` 几何槽（供空间索引/沿边索引用），
    读值留给 `project_assigned`。
    """
    out: List[Claim] = []
    if not isinstance(doc, dict):
        return out
    for comp in (doc.get("components") or []):
        if not isinstance(comp, dict):
            continue
        cid = str(comp.get("id", ""))
        out.append(Claim(slot=f"{K_COMP}:{cid}", value=None,
                         score=float(comp.get("score", 0.0) or 0.0),
                         source=str(comp.get("source", "stage1") or "stage1"), by=by,
                         region=_box4(comp.get("bbox_img")),
                         evidence={"type": comp.get("type"), "key": comp.get("key")},
                         attrs={"cid": cid, "type": comp.get("type")}))
    for cid, pts in (doc.get("pins") or {}).items():
        for i, p in enumerate(pts or []):
            out.append(Claim(slot=f"{K_PIN}:{cid}:{i}", value=None, score=0.0,
                             source="pintrace", by=by,
                             region=_box4([p[0] - 1.0, p[1] - 1.0, p[0] + 1.0, p[1] + 1.0]),
                             attrs={"cid": str(cid), "i": i, "x": p[0], "y": p[1]}))
    for cid, cols in (doc.get("pincol") or {}).items():
        for rec in (cols or []):
            if not isinstance(rec, dict):
                continue
            i = int(rec.get("idx", rec.get("i", 0)) or 0)
            side = str(rec.get("side", ""))
            along = rec.get("x") if side in ("L", "R") else rec.get("y")
            out.append(Claim(slot=f"{K_PIN}:{cid}:{rec.get('idx', i)}", value=None, score=0.0,
                             source="pincol", by=by,
                             region=_box4([float(rec.get("x", 0)) - 1.0,
                                           float(rec.get("y", 0)) - 1.0,
                                           float(rec.get("x", 0)) + 1.0,
                                           float(rec.get("y", 0)) + 1.0]),
                             attrs={"cid": str(cid), "i": i, "edge": side,
                                    "along": along, "pitch": rec.get("pitch"),
                                    "x": rec.get("x"), "y": rec.get("y"),
                                    "synth": rec.get("synth")}))
    for net in (doc.get("nets") or []):
        if not isinstance(net, dict):
            continue
        nid = str(net.get("id", ""))
        members = tuple(f"{c}[{i}]" for c, i in (net.get("pins") or []))
        out.append(Claim(slot=f"{K_NET}:{nid}", value=members,
                         score=float(net.get("n_seg", 0) or 0) / 10.0,
                         source="stage1:nets", by=by,
                         evidence={"n_seg": net.get("n_seg")}))
    return out


def project_assigned(assigned: dict, by: str = S2) -> List[Claim]:
    """stage2 `_assigned/<case>.json` → `Claim`：位号候选、型号、引脚号/名、网络标签。

    真实形状（case 5 实测）：`comps[cid] = {designator:{text,canon,conf,box,d,fit,box_i},
    value, partname:{text,conf,box,d}, pins:{"0":{number:{text,conf,box}, name:{...}}}}`
    —— `pins` 是**下标字符串 → 角色**的两层字典，不是列表。
    """
    out: List[Claim] = []
    if not isinstance(assigned, dict):
        return out
    for cid, alts in (assigned.get("desig_slots") or {}).items():
        out.extend(_claims_of_desig(str(cid), alts, by))
    for sid, alts in (assigned.get("struct_slots") or {}).items():
        for a in (alts or []):
            if not isinstance(a, dict):
                continue
            out.append(Claim(slot=f"{K_PIN_NUM}:{sid}", value=a.get("number", a.get("canon")),
                             score=float(a.get("score", 0.0) or 0.0),
                             source=str(a.get("source", "struct") or "struct"), by=by,
                             evidence={k: a.get(k) for k in ("text", "conf", "d")},
                             region=_box4(a.get("box"))))
    comps = assigned.get("comps") or {}
    items = comps.items() if isinstance(comps, dict) else enumerate(comps)
    for cid0, comp in items:
        if not isinstance(comp, dict):
            continue
        cid = str(comp.get("id", cid0))
        des = comp.get("designator")
        if isinstance(des, dict) and des.get("canon") not in (None, ""):
            out.append(Claim(slot=f"{K_DESIG}:{cid}", value=des.get("canon"),
                             score=1.0, source="stage2:designator", by=by,
                             evidence={k: des.get(k) for k in ("text", "conf", "d", "box_i", "fit")},
                             region=_box4(des.get("box")),
                             attrs={"raw_text": des.get("text"), "cid": cid}))
        part = comp.get("partname")
        if isinstance(part, dict) and part.get("text"):
            out.append(Claim(slot=f"{K_PART}:{cid}", value=part.get("text"),
                             score=float(part.get("conf", 0.0) or 0.0),
                             source="stage2:partname", by=by,
                             evidence={k: part.get(k) for k in ("conf", "d")},
                             region=_box4(part.get("box")), attrs={"cid": cid}))
        pins = comp.get("pins") or {}
        pit = pins.items() if isinstance(pins, dict) else enumerate(pins)
        for key, p in pit:
            if not isinstance(p, dict):
                continue
            for role, kind in (("number", K_PIN_NUM), ("name", K_PIN_NAME)):
                rec = p.get(role)
                if not isinstance(rec, dict) or rec.get("text") in (None, ""):
                    continue
                out.append(Claim(slot=f"{kind}:{cid}:{key}", value=rec.get("text"),
                                 score=float(rec.get("conf", 0.0) or 0.0),
                                 source=f"stage2:{role}", by=by,
                                 evidence={"conf": rec.get("conf"), "box": rec.get("box")},
                                 region=_box4(rec.get("box")),
                                 attrs={"cid": cid, "i": key, "raw_text": rec.get("text")}))
    # ★ 2026-10：`_assigned` 的 `net_labels` 已从 dict 改成**列表**（每项
    #   `{text, conf, box, d}`）。旧代码只吃 `.items()` ⇒ 本层自那次改动起
    #   **一张图都跑不起来**（`AttributeError: 'list' object has no attribute 'items'`），
    #   所以「从没 A/B 过」是字面意思。两种形状都接，避免下一次改形再静默弄坏。
    _nl = assigned.get("net_labels") or {}
    _nl_items = _nl.items() if isinstance(_nl, dict) else enumerate(_nl)
    for nid, lab in _nl_items:
        _v = lab.get("text") if isinstance(lab, dict) else lab
        if _v in (None, ""):
            continue
        out.append(Claim(slot=f"{K_NET}:{nid}", value=_v,
                         score=float(lab.get("conf", 1.0) or 0.0) if isinstance(lab, dict) else 1.0,
                         source="stage2:net_labels", by=by,
                         region=_box4(lab.get("box")) if isinstance(lab, dict) else None))
    return out


def load_case(arm_dir: str, case: str, ledger: Optional[Ledger] = None) -> Registry:
    """从一次运行的产物目录读一张图：stage1 + stage2 的 `_assigned`（只读）。"""
    reg = Registry(case=str(case), ledger=ledger)
    p1 = os.path.join(arm_dir, "runs", "stage1", f"{case}.stage1.json")
    pa = os.path.join(arm_dir, "runs", "stage2", "_assigned", f"{case}.json")
    if os.path.exists(p1):
        with open(p1, encoding="utf-8") as f:
            reg.add_many(project_stage1(json.load(f)))
    if os.path.exists(pa):
        with open(pa, encoding="utf-8") as f:
            reg.add_many(project_assigned(json.load(f)))
    return reg


# --------------------------------------------------------------------------- 内置校验
def check_desig_unique(reg: Registry, slot: str) -> List[Verdict]:
    """位号唯一性：同一个位号被两颗元件同时声称 ⇒ 低分者降权（不删）。"""
    sv = reg.slot(slot)
    if sv is None or sv.best is None:
        return []
    others = [s for s in reg.value_slots(sv.best.value) if s != slot]
    if not others:
        return [Verdict(slot=slot, kind="keep", before=sv.best.score,
                        after=sv.best.value, rule="位号唯一", by="unified:desig_unique",
                        at=S2)]
    best_other = max(float(reg.slot(s).best.score) for s in others if reg.slot(s).best)
    if float(sv.best.score) >= best_other:
        return [Verdict(slot=slot, kind="keep", before=sv.best.score, after=sv.best.value,
                        rule=f"撞号但分最高（{best_other:.3f}）", by="unified:desig_unique", at=S2)]
    return [Verdict(slot=slot, kind="soft", before=sv.best.score, after=sv.best.value,
                    rule=f"位号 {sv.best.value} 与 {len(others)} 个槽位撞号且分更低",
                    by="unified:desig_unique", at=S2, detail={"others": others})]


def check_pin_needs_name(reg: Registry, slot: str) -> List[Verdict]:
    """引脚必须「号 + 名」成对（`STAGE2_BOX_PIN_NEED_NAME` 的统一表达）。

    只有号没有名的脚，在 `stage2/assemble.py:1144` 那条 `num_of` 过滤下会被整个丢掉
    （5.13 的机制 A）⇒ 这里把它标出来，而不是让它静默消失。
    """
    if not slot.startswith(f"{K_PIN_NUM}:"):
        return []
    cid_i = slot.split(":", 1)[1]
    sv, sname = reg.slot(slot), reg.slot(f"{K_PIN_NAME}:{cid_i}")
    if sv is None or sv.best is None or sv.best.value in (None, ""):
        return []
    if sname is not None and sname.best is not None and sname.best.value not in (None, ""):
        return [Verdict(slot=slot, kind="keep", before=sv.best.score, after=sv.best.value,
                        rule="号名成对", by="unified:pin_pair", at=S2)]
    return [Verdict(slot=slot, kind="hard", before=sv.best.score, after=sv.best.value,
                    rule="有脚号但没有脚名 ⇒ 会被 num_of 过滤整脚丢掉",
                    by="unified:pin_pair", at=S2)]


def check_region_anchored(reg: Registry, slot: str) -> List[Verdict]:
    """取景框必须与候选来源贴住：没有框的候选（远端抢读）降权而非丢弃。"""
    sv = reg.slot(slot)
    if sv is None or sv.best is None:
        return []
    b = sv.best.region or sv.region
    if b is None:
        return [Verdict(slot=slot, kind="soft", before=sv.best.score, after=sv.best.value,
                        rule="候选没有取景框（无法复核来源）", by="unified:region", at=S2)]
    d = float((sv.best.evidence or {}).get("d", 0.0) or 0.0)
    if d and d > 40.0:
        return [Verdict(slot=slot, kind="soft", before=sv.best.score, after=sv.best.value,
                        rule=f"取景框离目标 {d:.1f}px（远端抢读）", by="unified:region", at=S2)]
    return [Verdict(slot=slot, kind="keep", before=sv.best.score, after=sv.best.value,
                    rule="取景框贴住目标", by="unified:region", at=S2)]


def check_value_spelling(reg: Registry, slot: str) -> List[Verdict]:
    """读到的值 vs 框里的原始文字：逐字不一致（`VCOUT`/`VOUT` 这类）标成 weak。

    注意：**只标不判**。谁对谁错要靠别的证据（`expect.py` 的词表、邻脚名的一致性），
    本层负责的是「把差异留痕」，而不是替它们裁决。
    """
    sv = reg.slot(slot)
    if sv is None or sv.best is None:
        return []
    raw = (sv.best.attrs or {}).get("raw_text") or (sv.best.evidence or {}).get("text")
    val = sv.best.value
    if not raw or val is None or str(raw) == str(val):
        return []
    return [Verdict(slot=slot, kind="soft", before=sv.best.score, after=val,
                    rule=f"读法与框内文字不一致（raw={raw!r} -> {val!r}）",
                    by="unified:spelling", at=S2)]


def check_desig_near_comp(reg: Registry, slot: str) -> List[Verdict]:
    """位号取景框必须贴住**它自己那颗**元件：用空间索引找最近的元件框。

    这是「位号↔元件」这条最关键绑定（`docs/RESULTS.md:842` 说真瓶颈就是位号绑定）
    的最小可复算判据 —— 不依赖 OCR 分数，只看几何。走 `GridIndex`，代价与邻近格数
    成正比而不是与槽位数成正比。
    """
    if not slot.startswith(f"{K_DESIG}:"):
        return []
    cid = slot.split(":", 1)[1]
    sv = reg.slot(slot)
    if sv is None or sv.best is None or sv.best.region is None:
        return []
    near = reg.near(sv.best.region, DESIG_NEAR, kinds=(K_COMP,))
    sc = float(sv.best.score)
    if not near:
        return [Verdict(slot=slot, kind="soft", before=sc, after=sv.best.value,
                        rule=f"{DESIG_NEAR:.0f}px 内没有任何元件框（位号是远端抢读？）",
                        by="unified:desig_near_comp", at=S2)]
    if near[0] != f"{K_COMP}:{cid}":
        return [Verdict(slot=slot, kind="soft", before=sc, after=sv.best.value,
                        rule=f"位号框最贴的是 {near[0]}（不是自己 {K_COMP}:{cid}）",
                        by="unified:desig_near_comp", at=S2, detail={"near": near[:3]})]
    return [Verdict(slot=slot, kind="keep", before=sc, after=sv.best.value,
                    rule="位号框贴住所属元件", by="unified:desig_near_comp", at=S2)]


def check_pin_column_order(reg: Registry, slot: str) -> List[Verdict]:
    """引脚列：`idx` 与**沿边坐标**必须同序（否则就是 5.13「重启编号」所在的那种域）。

    只用沿边索引看**紧邻的两个**槽位（`bisect`），代价 O(log n)，与列长无关。
    """
    if not slot.startswith(f"{K_PIN}:"):
        return []
    sv = reg.slot(slot)
    if sv is None or not sv.attrs.get("edge") or sv.attrs.get("along") is None:
        return []
    ai = reg.along_index()
    nb = ai.neighbors(slot, sv.attrs, k=1)
    if not nb:
        return []
    mine = sv.attrs.get("i")
    near = reg.slot(nb[0])
    if near is None or mine is None:
        return []
    other = near.attrs.get("i")
    if other is None:
        return []
    if (mine < other) != (float(sv.attrs["along"]) < float(near.attrs.get("along", 0.0))):
        return [Verdict(slot=slot, kind="soft", before=1.0, after=f"{sv.attrs.get('edge')}{mine}",
                        rule=f"沿边顺序与 idx 反序（{mine} vs {other}）⇒ 单调剪枝会误杀",
                        by="unified:pin_column", at=S1, detail={"neighbor": nb[0]})]
    return [Verdict(slot=slot, kind="keep", before=1.0, after=f"{sv.attrs.get('edge')}{mine}",
                    rule="沿边顺序与 idx 同序", by="unified:pin_column", at=S1)]


def default_checks() -> List[Check]:
    """本层默认登记的校验（全部只读、只留痕）。"""
    return [
        Check("desig_unique", check_desig_unique, kinds=(K_DESIG,), by="unified",
              at=S2, docs="位号唯一性（同号低分者降权）"),
        Check("desig_near_comp", check_desig_near_comp, kinds=(K_DESIG,), by="unified",
              at=S2, docs="位号框必须贴住所属元件（走空间索引）"),
        Check("pin_pair", check_pin_needs_name, kinds=(K_PIN_NUM,), by="unified",
              at=S2, docs="脚号必须有脚名，否则整脚被丢"),
        Check("pin_column", check_pin_column_order, kinds=(K_PIN,), by="unified",
              at=S1, docs="引脚列的 idx 应与沿边坐标同序（走沿边索引）"),
        Check("region", check_region_anchored, kinds=(K_DESIG, K_PIN_NUM, K_PIN_NAME), by="unified",
              at=S2, docs="取景框必须贴住目标"),
        Check("spelling", check_value_spelling, kinds=(K_PIN_NAME, K_DESIG), by="unified",
              at=S2, docs="读法与框内文字不一致只标不判"),
    ]


# --------------------------------------------------------------------------- 自检 / CLI
def _selftest() -> int:
    ok, n = [], 0

    def chk(cond, name):
        nonlocal n
        n += 1
        ok.append(bool(cond))
        print(("  [ok] " if cond else "  [FAIL] ") + name)

    # 1. 框距离
    chk(abs(box_dist((0, 0, 10, 10), (14, 0, 20, 10)) - 4.0) < 1e-9, "box_dist 横向间隙")
    chk(box_dist((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0, "box_dist 相交为 0")
    chk(abs(point_box_dist((0, 0), (3, 4, 10, 10)) - 5.0) < 1e-9, "point_box_dist 对角")

    # 2. 注册表 + 索引
    reg = Registry("t")
    reg.add(Claim(slot="desig:c1", value="R1", score=0.9, region=(0, 0, 10, 10)))
    reg.add(Claim(slot="desig:c1", value="R2", score=0.4, region=(0, 0, 10, 10)))
    reg.add(Claim(slot="desig:c2", value="R1", score=0.5, region=(100, 100, 110, 110)))
    chk(reg.slot("desig:c1").best.value == "R1", "多候选按分排序")
    chk(reg.slot("desig:c1").margin == 0.5, "分差 = 最优-次优")
    chk(reg.slot("desig:c1").ambiguous(0.6), "分差小于 MARGIN ⇒ ambiguous")
    chk([k for k, _s in reg.conflicts()] == ['"R1"'], "撞号冲突一次算清")
    chk(reg.value_slots("R1") == ["desig:c1", "desig:c2"], "取值反查槽位")
    chk(reg.near((12, 0, 13, 1), 10.0) == ["desig:c1"], "空间索引按距离排序")
    chk(reg.near((1000, 1000, 1001, 1001), 5.0) == [], "远处查不到")
    chk(len(reg.slots(K_DESIG)) == 2 and len(reg.slots()) == 2, "按种类取槽位")

    # 3. 空间索引 = 暴力枚举（随机 200 个框）
    import random
    random.seed(7)
    reg2 = Registry("g")
    for i in range(200):
        x, y = random.uniform(0, 800), random.uniform(0, 600)
        reg2.add(Claim(slot=f"desig:g{i}", value=f"U{i}", score=1.0,
                       region=(x, y, x + random.uniform(2, 40), y + random.uniform(2, 40))))
    q = (300.0, 200.0, 330.0, 230.0)
    fast = set(reg2.near(q, 25.0))
    brute = {s for s, sv in reg2.by_slot.items() if box_dist(q, sv.region) <= 25.0}
    chk(fast == brute, f"GridIndex 与暴力枚举一致（{len(fast)} 命中）")

    # 4. 沿边索引
    reg3 = Registry("a")
    for i, (x, y) in enumerate([(10.0, 100.0), (10.0, 80.0), (10.0, 120.0)]):
        reg3.add(Claim(slot=f"pin_number:c1:{i}", value=str(i + 1), score=1.0,
                       attrs={"cid": "c1", "edge": "L", "along": y}))
    ai = reg3.along_index()
    chk(ai.neighbors("pin_number:c1:1", {"cid": "c1", "edge": "L", "along": 100.0}, 1)
        == ["pin_number:c1:0"], "沿边最近邻（平票取坐标小的）")
    chk(set(ai.neighbors("pin_number:c1:1", {"cid": "c1", "edge": "L", "along": 100.0}, 2))
        == {"pin_number:c1:0", "pin_number:c1:2"}, "沿边最近 2 个（按坐标序，不是插入序）")
    chk(ai.rank("pin_number:c1:0", {"cid": "c1", "edge": "L", "along": 80.0}) == 0, "沿边序位")

    # 5. 校验 + 留痕（Verdict → 既有账本）
    reg4 = Registry("v", ledger=Ledger("v"))
    reg4.add(Claim(slot="pin_number:c9:0", value="3", score=0.9, region=(5, 5, 6, 6)))
    reg4.add(Claim(slot="desig:c9", value="C1", score=0.8, region=(5, 5, 9, 9)))
    reg4.register(*default_checks())
    vs = reg4.run()
    chk(any(v.kind == "hard" and v.slot == "pin_number:c9:0" for v in vs), "缺脚名 ⇒ hard")
    chk(any(v.kind == "keep" for v in vs), "合格也留痕（keep）")
    chk(reg4.ledger.n_demotions == sum(1 for v in vs if v.kind != "keep"), "降权条数 = 非 keep 的判定数")
    chk(all(d.score_after > 0.0 for d in reg4.ledger.demotions), "降权分**永不取 0**")
    chk(reg4.ledger.stats()["n_events"] == len(vs), "事件数 = 判定数（含 keep）")

    # 6. 议程：只挑需要复核的槽位
    ag = Agenda.from_registry(reg, margin=0.6)
    chk(set(ag.pending()) == {"desig:c1"}, "议程只含分差小的槽位（c2 单候选不进）")
    chk(ag.pop() == "desig:c1" and len(ag) == 0, "议程按优先级弹出")

    # 7. 投影：三种历史形状 → 同一套 Claim
    class _Alt:
        def __init__(s, v, sc, src="x"):
            s.value, s.score, s.source, s.evidence = v, sc, src, {}
    class _Slot:
        def __init__(s, alts):
            s.alts, s.context, s.decided_by = alts, {}, None
    class _HY:
        slots = {"desig:k1": _Slot([_Alt("U1", 1.15), _Alt("U580", 0.825)])}
    class _Pool:
        hy = _HY()
        regions = {}
    cs = project_pool(_Pool())
    chk(len(cs) == 2 and cs[0].slot == "desig:k1" and cs[0].value == "U1", "ClaimPool → Claim")
    cs2 = project_pool({"c1": [{"canon": "R5", "text": "R5", "score": 0.7, "box": [1, 2, 3, 4]}]})
    chk(cs2[0].slot == "desig:c1" and cs2[0].region == (1.0, 2.0, 3.0, 4.0), "desig_all 形状 → Claim")
    cs3 = project_stage1({"pins": {"c1": [[10.0, 20.0]]},
                          "components": [{"id": "c1", "type": "box", "score": 0.9,
                                          "bbox_img": [0, 0, 50, 50]}],
                          "nets": [{"id": "net_1", "pins": [["c1", 0]], "n_seg": 3}]})
    chk(any(c.slot == "net:net_1" and c.value == ("c1[0]",) for c in cs3), "stage1 nets → Claim")
    chk(any(c.slot == "comp:c1" and c.region == (0.0, 0.0, 50.0, 50.0) for c in cs3),
        "stage1 components → comp 槽（带框）")
    cs4 = project_assigned({"desig_slots": {"c0": [{"canon": "U2", "score": 0.67}]},
                            "comps": {"c0": {"designator": {"text": "U2", "canon": "U2", "box": [1, 1, 9, 9]},
                                             "partname": {"text": "TPSW", "conf": 0.85},
                                             "pins": {"0": {"number": {"text": "1", "conf": 0.99},
                                                            "name": {"text": "GND", "conf": 0.87}}}}}})
    chk({c.slot for c in cs4} == {"desig:c0", "partname:c0", "pin_number:c0:0", "pin_name:c0:0"},
        "assigned（位号+型号+号+名）→ Claim")
    chk(sum(1 for c in cs4 if c.slot == "desig:c0") == 2, "同一槽位的两个来源各自成条（未合并前）")

    # 7b. 同值多来源 ⇒ 合并（取最高分、证据并起来）
    reg6 = Registry("m")
    reg6.add(Claim(slot="desig:c1", value="U1", score=0.8, source="a"))
    reg6.add(Claim(slot="desig:c1", value="U1", score=1.15, source="b"))
    sv6 = reg6.slot("desig:c1")
    chk(len(sv6.claims) == 1 and sv6.best.score == 1.15, "同值多来源合并并取最高分")
    chk(sv6.best.evidence["also_from"][0]["source"] == "b", "合并时留下另一个来源")

    # 8. 适配器：旧的布尔判据 / 裸 dict 判据都能包进来
    ck1 = Check.predicate(lambda v: v == "GND", name="is_gnd", kinds=(K_PIN_NAME,),
                          fail_kind="soft", rule="脚名不是 GND")
    reg5 = Registry("p")
    reg5.add(Claim(slot="pin_name:c1:0", value="VCOUT", score=0.8))
    reg5.register(ck1)
    chk(reg5.run()[0].kind == "soft", "Check.predicate 包旧布尔判据")
    ck2 = Check.adapt(lambda r, s: [{"kind": "trunc", "rule": "超预算"}], name="legacy")
    reg5.register(ck2)
    chk(any(v.rule == "超预算" for v in reg5.run(only=["legacy"])), "Check.adapt 包裸 dict 判据")

    # 9. 几何校验用空间索引：位号贴自己那颗元件 / 贴错元件
    reg7 = Registry("geo")
    reg7.add(Claim(slot="comp:c1", value="box", score=0.9, region=(0, 0, 50, 50)))
    reg7.add(Claim(slot="comp:c2", value="box", score=0.9, region=(400, 400, 450, 450)))
    reg7.add(Claim(slot="desig:c1", value="U1", score=0.9, region=(20, 20, 30, 30)))
    reg7.add(Claim(slot="desig:c2", value="U2", score=0.9, region=(30, 30, 40, 40)))
    reg7.register(*default_checks())
    r7 = {v.slot: v for v in reg7.run(only=["desig_near_comp"])}
    chk(r7["desig:c1"].kind == "keep", "位号贴自己那颗元件 ⇒ keep")
    chk(r7["desig:c2"].kind == "soft" and "comp:c1" in r7["desig:c2"].rule,
        "位号更贴另一颗元件 ⇒ soft 且点出是谁")
    chk(reg7.index().n_query >= 2, "几何校验确实走了空间索引")

    # 10. 沿边校验用沿边索引：idx 与沿边坐标同序 / 反序
    reg8 = Registry("col")
    reg8.add(Claim(slot="pin:c1:0", value=None, attrs={"cid": "c1", "i": 0, "edge": "R", "along": 80.0}))
    reg8.add(Claim(slot="pin:c1:1", value=None, attrs={"cid": "c1", "i": 1, "edge": "R", "along": 100.0}))
    reg8.add(Claim(slot="pin:c1:2", value=None, attrs={"cid": "c1", "i": 2, "edge": "R", "along": 120.0}))
    reg8.add(Claim(slot="pin:c1:3", value=None, attrs={"cid": "c1", "i": 0, "edge": "L", "along": 300.0}))
    reg8.add(Claim(slot="pin:c1:4", value=None, attrs={"cid": "c1", "i": 1, "edge": "L", "along": 280.0}))
    reg8.register(*default_checks())
    r8 = {v.slot: v for v in reg8.run(only=["pin_column"])}
    chk(r8["pin:c1:0"].kind == "keep" and r8["pin:c1:2"].kind == "keep",
        "idx 与沿边坐标同序 ⇒ keep（整列）")
    chk(r8["pin:c1:3"].kind == "soft" and "反序" in r8["pin:c1:3"].rule,
        "idx 与沿边坐标反序 ⇒ soft（5.13 重启编号那种域）")

    # 11. 撞号分域：位号全局唯一，脚号只在一颗元件内唯一
    reg9 = Registry("uniq")
    reg9.add(Claim(slot="pin_number:c1:0", value="2"))
    reg9.add(Claim(slot="pin_number:c2:0", value="2"))
    chk(len(reg9.conflicts()) == 1 and reg9.conflicts(UNIQUE_KINDS) == [],
        "脚号跨元件重复 ⇒ 不算唯一性撞号")
    reg9.add(Claim(slot="desig:c1", value="U1"))
    reg9.add(Claim(slot="desig:c2", value="U1"))
    chk(len(reg9.conflicts(UNIQUE_KINDS)) == 1, "位号重复 ⇒ 唯一性撞号")
    chk(reg9.summary()["n_uniq_conflicts"] == 1 and reg9.summary()["n_conflicts"] == 2,
        "summary 同时报裸撞号与唯一性撞号")

    # 12. 整池审计适配器（`fusion/plausible.check` 那种「跑一遍整池、返回 issue 清单」）
    regA = Registry("issues")
    regA.add(Claim(slot="desig:c1", value="U1", score=0.8))
    regA.add(Claim(slot="desig:c2", value="U1", score=0.6))
    regA.add(Claim(slot="desig:c3", value="U2", score=0.9))
    calls = {"n": 0}

    def fake_check(_reg):
        calls["n"] += 1
        return [{"kind": "key_duplicate", "severity": 0.8,
                 "why": "位号 U1 被 2 个槽位同时占用", "need_llm": True,
                 "slots": ["comp.c1.key", "comp.c2.key"]},
                {"kind": "weak", "severity": 0.2, "why": "弱信号，只报不判",
                 "slots": ["comp.c3.key"]}]

    regA.register(Check.from_issues(
        fake_check, name="plausible", kinds=(K_DESIG,),
        map_slot=lambda s: s.replace("comp.", "desig:").replace(".key", "")))
    va = regA.run()
    slotv: Dict[str, List[Verdict]] = {}
    for v in va:
        slotv.setdefault(v.slot, []).append(v)
    chk(slotv["desig:c1"][0].kind == "hard" and "U1" in slotv["desig:c1"][0].rule,
        "整池审计：severity 0.8 ⇒ hard，why 记成理由")
    chk(slotv["desig:c3"][0].kind == "soft", "整池审计：severity 0.2 ⇒ soft")
    chk(calls["n"] == 1, "整池审计每个注册表只跑一次（不是每槽位跑一遍）")
    chk(all(v.slot.startswith("desig:") for v in va), "map_slot 把旧槽位名翻成统一槽位名")

    print("自检 %d 项：%s" % (n, "全部通过" if all(ok) else "有 %d 项失败" % ok.count(False)))
    return 0 if all(ok) else 1


def _demo(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="统一交叉校验层（只读诊断）")
    ap.add_argument("--dir", default="", help="一次运行的产物目录（含 runs/stage1 与 runs/stage2）")
    ap.add_argument("--case", default="", help="图号")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.dir or not a.case:
        return _selftest()
    reg = load_case(a.dir, a.case, ledger=Ledger(str(a.case)))
    reg.register(*default_checks())
    ag = Agenda.from_registry(reg, margin=a.margin)
    vs = reg.run()
    if a.json:
        d = reg.to_json(a.top)
        d["agenda"] = ag.pending()[:a.top]
        d["verdicts"] = [asdict(v) for v in vs[:a.top]]
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return 0
    s = reg.summary()
    print("case %s @ %s" % (a.case, a.dir))
    print("  候选 %d 条 / 槽位 %d 个 / 取值 %d 个  %s" % (s["n_claims"], s["n_slots"],
                                                         s["n_vals"], s["by_kind"]))
    print("  校验 %d 条 → 判定 %d 条（其中降权 %d）  撞号 %d（唯一性域 %d）  需复核 %d  空间查询 %d"
          % (s["n_checks"], s["n_verdicts"],
             sum(1 for v in vs if v.kind != "keep"), s["n_conflicts"],
             s["n_uniq_conflicts"], s["n_ambiguous"], s["n_grid_query"]))
    print("  议程（分差小的优先）：")
    for slot in ag.pending()[:a.top]:
        sv = reg.slot(slot)
        b, s2 = sv.best, sv.second
        print("    %-26s %-10s %.3f vs %.3f  (差 %.3f)"
              % (slot, str(b.value)[:10], b.score, (s2.score if s2 else 0.0), sv.margin))
    # 默认只列**唯一性域**（位号）撞号：脚号/脚名跨元件重复是正常的（见 `UNIQUE_KINDS`）。
    conf = reg.conflicts(UNIQUE_KINDS)
    if conf:
        print("  撞号（唯一性域 = 位号）：")
        for k, slots in conf[:a.top]:
            print("    %-12s %s" % (k, " ".join(slots)))
    elif s["n_conflicts"]:
        print("  撞号：唯一性域 0 条（其余 %d 条都是脚号/脚名跨元件重复，正常）"
              % s["n_conflicts"])
    bad = [v for v in vs if v.kind != "keep"]
    if bad:
        print("  降权（前 %d 条）：" % min(len(bad), a.top))
        for v in bad[:a.top]:
            print("    %-9s %-24s %.3f  %s" % (v.kind, v.slot, float(v.before or 0.0), v.rule))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_demo(sys.argv[1:]))
