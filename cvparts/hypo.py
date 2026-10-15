"""多假设槽位（Slot / Alternative）—— 「判不出来就并行存几种，最后让 LLM 选」。

为什么需要它
------------
`testpoint` 一个类有三种画法、`gnd` 一个类有十种模式、位号 `R1/R2/…` 与元件类别之间是
**带约束的全局指派**、`net_input` 的图章经常压不到任何东西。这些位置**规则层只能给概率，
不该硬猜**：猜错 = 多出一个错误元件，比「留白」更亏（官方按 key 匹配，猜错既是 fp 又占掉
一个名额）。

所以整条管线的每一次「有歧义的判断」都产出一个 **Slot**：一个决策点 + 若干候选
**Alternative**（各自带 score / 来源 / 证据）。规则层能定就定（`resolve`），定不了就
**原样保留全部候选**；后续由
  ①几何/上下文规则，
  ②`textrec` 的文字识别与绑定，
  ③LLM（结合上下文在候选里选）
逐层消解。最终输出前用 `collapse_types()` 把子类型折回官方类名，保证评测口径不变。

数据形状（可直接 json.dump，与赛题要求的 `decision_log.json` 对齐）
------------------------------------------------------------------
```json
{
  "case": "0050",
  "slots": {
    "t0007.type": {
      "question": "t0007 这个符号是什么元件？",
      "alts": [
        {"value": "testpoint", "score": 0.85, "source": "cv:ring3",
         "evidence": {"n_rings": 3, "r_max": 20.1}},
        {"value": "testpoint", "score": 0.55, "source": "cv:hole",
         "evidence": {"r": 5.0, "circ": 0.90}}
      ],
      "context": {"cx": 2996.0, "cy": 212.0, "near_text": ["TP1"]},
      "chosen": null, "decided_by": null
    }
  }
}
```

用法
----
```python
h = CaseHypotheses("0050")
h.add("t0007.type", "testpoint", 0.85, source="cv:ring3",
      evidence={"n_rings": 3}, question="t0007 是什么元件？",
      context={"near_text": ["TP1"]})
h.add("t0007.type", "testpoint", 0.55, source="cv:hole", evidence={"r": 5.0})
h.resolve("t0007.type", "testpoint", by="rule:agree")   # 两个候选同值 -> 直接定
for s in h.unresolved():                                 # 还没定的，交给 LLM
    ...
```
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# 歧义判据：最优与次优的差距小于这个值就算「判不出来」
MARGIN = 0.15
# ★ 被改写掉的旧值的降权分（见 `CaseHypotheses.resolve`）：不删、只削弱。
OVERWRITE_DEMOTE = 0.05
# 候选上限（防止一个槽位塞进几十个候选把 LLM 上下文撑爆）
MAX_ALTS = 6

# ---------------------------------------------------------------- 不删除：降权档位
# owner（m23945）的规矩：「可以削弱内容的置信度，但不能轻易覆盖」「任何『距离/分数
# 超阈值就丢弃』的判断都是错的，超限项必须留在系统里，附一个可比较的标量，由下游
# 降权或复核」（`docs/reports/工程哲学.md:41`、`:94`）。
# 统一档位（**任何**「本来会被丢掉」的东西都落到这四档之一，而不是 0 或被删）：
DEMOTE_OVERWRITE = OVERWRITE_DEMOTE   # 0.05  被别的证据改写
DEMOTE_HARD = 0.05                    # 0.05  判据被强证据否掉（原来是 0.0 = 删除）
DEMOTE_SOFT = 0.15                    # 0.15  与主流不一致（弱判据）
DEMOTE_TRUNC = 0.02                   # 0.02  被截断/挤出（超上限、超预算）
# 详细记录的条数上限；**超出的只丢明细不丢计数**（`stats()["truncated_demotions"]`），
# 所以「有多少东西被降权过」永远可查，只是不再逐条可点名。
LEDGER_DETAIL_LIMIT = 600


@dataclass
class Demotion:
    """一条「本来会被丢掉、现在只降权」的记录。

    `score_before > 0` 且 `score_after < score_before` 就是一次削弱；`score_after`
    取上面四档之一，绝不取 0 —— 0 在很多下游等于「不可选」，那就又变成删除了。
    """
    where: str = ""                 # 槽位 id / 引脚下标 / 分组名
    value: Any = None
    score_before: float = 0.0
    score_after: float = 0.0
    reason: str = ""
    by: str = ""
    at: str = ""                    # 阶段标记，如 "stage1:pincol"、"stage2:structmatch"

    @property
    def lost(self) -> float:
        return round(self.score_before - self.score_after, 6)


@dataclass
class Event:
    """merge / split / demote / drop / truncate / overwrite 的统一事件（时间维度）。

    `before` / `after` 是**事实**（不是差集）：合并前后各有哪些，拆分前后各有哪些，
    谁来做的（`by`）、为什么（`reason`）。owner 要的「本来不是同一个结果变成同一个、
    本来是同一个但被拆分，这些都有意义」就落在这里。
    """
    kind: str = ""
    where: str = ""
    before: list = field(default_factory=list)
    after: Any = None
    reason: str = ""
    by: str = ""
    at: str = ""


class Ledger:
    """全局「不删除只削弱」账本：降权记录 + 事件 + 重扫队列。

    * `demote(...)` 记一次削弱（值仍然在候选池里，只是分低了）；
    * `record(...)` 记一次 merge/split/drop/…（信息搬家而不是消失）；
    * `touched_slots()` = **重扫队列**：只有产生过事件的槽位才需要再扫一次
      （`工程哲学.md:94` 第 ③ 条），没出事的槽位不浪费算力；
    * `dump()` 是可 json 化的紧凑结构，直接塞进 stage1/stage2 产物给复核用。
    """

    def __init__(self, case: str = ""):
        self.case = case
        self.demotions: list = []
        self.events: list = []
        self.n_demotions = 0
        self.n_events = 0
        self.by_kind: dict = {}

    # ---- 写 ----
    def demote(self, where: str, value=None, score_before: float = 0.0,
               score_after: float = DEMOTE_SOFT, reason: str = "",
               by: str = "", at: str = "") -> None:
        self.n_demotions += 1
        if len(self.demotions) < LEDGER_DETAIL_LIMIT:
            self.demotions.append(Demotion(where=where, value=value,
                                           score_before=round(float(score_before), 4),
                                           score_after=round(float(score_after), 4),
                                           reason=reason, by=by, at=at))

    def record(self, kind: str, where: str, before=None, after=None,
               reason: str = "", by: str = "", at: str = "") -> None:
        self.n_events += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        if len(self.events) < LEDGER_DETAIL_LIMIT:
            self.events.append(Event(kind=kind, where=where,
                                     before=list(before or []), after=after,
                                     reason=reason, by=by, at=at))

    # ---- 读 ----
    def touched_slots(self) -> list:
        out = []
        for x in list(self.events) + list(self.demotions):
            w = getattr(x, "where", "")
            if w and w not in out:
                out.append(w)
        return out

    def stats(self) -> dict:
        return {"case": self.case, "n_events": self.n_events,
                "n_demotions": self.n_demotions,
                "by_kind": dict(self.by_kind),
                "strictly_positive": all(d.score_after > 0.0 for d in self.demotions),
                "truncated_events": max(0, self.n_events - len(self.events)),
                "truncated_demotions": max(0, self.n_demotions - len(self.demotions)),
                "n_rescan": len(self.touched_slots())}

    def dump(self, sample: int = 6) -> dict:
        """紧凑、可 json 化：计数 + 按档位汇总 + 少量样例 + 重扫队列。"""
        by_after: dict = {}
        lost = 0.0
        for d in self.demotions:
            k = round(float(d.score_after), 4)
            by_after[k] = by_after.get(k, 0) + 1
            lost += d.lost
        return {"stats": self.stats(),
                "by_score_after": by_after,
                "lost_score_total": round(lost, 4),
                "rescan": self.touched_slots()[:40],
                "events": [asdict(e) for e in self.events[:sample]],
                "demotions": [asdict(d) for d in self.demotions[:sample]]}

    def clear(self) -> None:
        self.demotions.clear(); self.events.clear()
        self.n_demotions = self.n_events = 0
        self.by_kind.clear()


# 进程内默认账本：任何模块都能 `from cvparts.hypo import LEDGER` 记账，不必层层传参。
# （stage1 走 ProcessPoolExecutor，所以每个 worker 各有一本，产物里带不回来 —— 见
#  `stage1/pipeline.py` 把 `LEDGER.dump()` 塞进 doc 的做法。）
LEDGER = Ledger()


def reset_ledger(case: str = "") -> Ledger:
    LEDGER.case = case
    LEDGER.clear()
    return LEDGER


@dataclass
class Alt:
    value: Any
    score: float = 0.0
    source: str = ""
    evidence: dict = field(default_factory=dict)

    def key(self) -> str:
        try:
            return json.dumps(self.value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return str(self.value)


@dataclass
class Slot:
    slot_id: str
    question: str = ""
    alts: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    chosen: Any = None
    decided_by: str | None = None
    # ★★ 「时间维度」的载体（owner m23945 原意：「本来不是同一个结果变成同一个、
    #   本来是同一个但被拆分 —— 这些内容都有意义，可作为优化指引的上下文」）。
    #   每次 `chosen` 被改写都往这里追加一条，**旧值永不丢失**。
    history: list = field(default_factory=list)
    # ★★ 「判据不能二值 / 信息只降权不销毁」的第 2 个落点：候选超上限时原来是一句
    #   静默的 `del self.alts[MAX_ALTS:]` —— 第 7 名以后的候选**直接不存在**，
    #   既没有计数也没有值，复核者永远不知道「本来还有别的可能」。
    #   现在超出上限的候选**移进 `overflow`**（保留值/分/来源），`alts`/`best()`
    #   的行为一字不变，变的只是「被挤出去的东西还在」。
    overflow: list = field(default_factory=list)

    # ---- 候选管理 ----
    def add(self, alt: Alt) -> None:
        for a in self.alts:
            if a.key() == alt.key():
                # 同一个值被多个来源支持 -> 取最高分并把证据合起来
                a.evidence.setdefault("also_from", []).append(
                    {"source": alt.source, "score": alt.score, "evidence": alt.evidence})
                # ★★ 取最高分之后**必须重排**：`best()` 是 `alts[0]`，而列表是按分
                # 降序维护的。原来这里 `a.score = max(...)` 之后直接 `return`，分数
                # 涨了、名次没动 —— 于是 `best()` 会返回**分数更低**的那个候选。
                # 实测（`work\tools\diag\_shapekeys2.py`，case 0001 的
                # `comp.c0003.key`）：`U580`(0.825) 先入表排在第一，随后 VLM 的
                # `U1`(1.15) 合并进来把 `U1` 的分抬到 1.15，但 `alts[0]` 仍是
                # `U580` ⇒ `best()` 报 `U580`，**分更高的正确读法被埋掉**。
                a.score = max(a.score, alt.score)
                self.alts.sort(key=lambda a: -a.score)
                self._spill()
                return
        self.alts.append(alt)
        self.alts.sort(key=lambda a: -a.score)
        self._spill()

    def _spill(self) -> None:
        """把超出 `MAX_ALTS` 的候选**移进 `overflow`**（不删除、可追溯）。

        ★ 到 2026-10 为止这里只做了「搬家」，全局账本 `LEDGER` 里没有它的名字 ——
        复核者能看到 `overflow` 里有什么，却看不到「一共挤出过多少、挤出去多少分」。
        现在每次挤出都记一条 `truncate` 事件 + 一条 `DEMOTE_TRUNC` 降权
        （信息仍然在 `overflow` 里，只是被标成「被截断的」）。
        """
        if len(self.alts) > MAX_ALTS:
            gone = self.alts[MAX_ALTS:]
            self.overflow.extend(gone)
            del self.alts[MAX_ALTS:]
            for a in gone:
                LEDGER.demote(self.slot_id, a.value, a.score, DEMOTE_TRUNC,
                              reason=f"候选数超过 MAX_ALTS={MAX_ALTS}，移入 overflow",
                              by=a.source or "?", at="hypo:spill")
                LEDGER.record("truncate", self.slot_id, before=[a.value], after=None,
                              reason="超出候选上限（值保留在 overflow）",
                              by=a.source or "?", at="hypo:spill")

    def resolved(self) -> bool:
        return self.decided_by is not None

    def ambiguous(self, margin: float = MARGIN) -> bool:
        if self.resolved():
            return False
        return len(self.alts) > 1 and (self.alts[0].score - self.alts[1].score) < margin

    def best(self) -> Alt | None:
        return self.alts[0] if self.alts else None

    def to_dict(self) -> dict:
        return {"question": self.question,
                "alts": [asdict(a) for a in self.alts],
                "context": self.context,
                "chosen": self.chosen, "decided_by": self.decided_by,
                "history": list(self.history),
                "overflow": [asdict(a) for a in self.overflow]}


@dataclass
class CaseHypotheses:
    case: str
    slots: dict = field(default_factory=dict)
    # 子类型 -> 官方类名（输出前折叠用）
    folds: dict = field(default_factory=dict)

    # ---- 写 ----
    def add(self, slot_id: str, value: Any, score: float = 0.0, source: str = "",
            evidence: dict | None = None, question: str = "",
            context: dict | None = None) -> Slot:
        s = self.slots.get(slot_id)
        if s is None:
            s = Slot(slot_id=slot_id, question=question, context=dict(context or {}))
            self.slots[slot_id] = s
        elif context:
            s.context.update(context)
        s.add(Alt(value=value, score=score, source=source, evidence=dict(evidence or {})))
        return s

    def fold(self, subtype: str, official: str) -> None:
        """登记一条折叠规则：子类型 subtype 在输出时算作 official。"""
        self.folds[subtype] = official

    def fold_value(self, v):
        """把候选值折成官方类名。"""
        return self.folds.get(v, v)

    def resolve(self, slot_id: str, value: Any, by: str) -> None:
        """定下一个槽位的值。

        ★★ 这里是 owner「**可以削弱内容的置信度，但不能轻易覆盖，不然很多关键
        信息都被覆盖了**」的落点（原来只有两行 `s.chosen = value` /
        `s.decided_by = by`，被顶掉的值**立刻从世界上消失**，只在 `alts` 里以
        普通候选的身份偶然存活，`decided_by` 也没有任何历史）。

        泛化的规矩（不是特例）：**任何一次「改写」都是一次有记录的迁移** ——
          * 旧值 + 旧的决定者 → 追加进 `s.history`（时间维度，供优化指引当上下文）；
          * 旧值重新入池为一条**降权候选** `OVERWRITE_DEMOTE`（仍然可被看见、
            可以被别的证据重新扶起来，但不会因为一次覆盖就彻底没了）。
        交付物读的仍然是 `chosen` / `decided_by`，所以**行为与分数不变**，
        变的只是「信息不再被销毁」。
        """
        s = self.slots.get(slot_id)
        if s is None:
            raise KeyError(slot_id)
        prev = s.chosen
        if prev is not None and str(prev) != str(value):
            src = s.decided_by or "?"
            s.history.append({"value": prev, "decided_by": s.decided_by,
                              "replaced_by": by})
            s.add(Alt(value=prev, score=OVERWRITE_DEMOTE, source=src,
                      evidence={"被覆盖": by, "降权理由": "改写只降权不删除"}))
            # ★ 全局账本：被顶掉的值进了 `history`（槽位本地）+ 一条 overwrite 事件
            #   与一条 `DEMOTE_OVERWRITE` 降权（跨槽位可查）。
            prev_score = 1.0
            for a in s.alts:
                if str(a.value) == str(prev):
                    prev_score = max(prev_score, float(a.score))
            LEDGER.demote(slot_id, prev, prev_score, DEMOTE_OVERWRITE,
                          reason=f"被 {value!r} 改写（{by}）", by=src, at="hypo:resolve")
            LEDGER.record("overwrite", slot_id, before=[prev], after=value,
                          reason="改写只降权不删除", by=by, at="hypo:resolve")
        s.chosen = value
        s.decided_by = by

    def auto_resolve(self) -> dict:
        """把「折叠后所有候选都是同一个官方类」的槽位直接定下来。

        这一步是「内部按本质不同的几种画法跑，输出时再合并回一个类」的落点：
        `testpoint` / `testpoint:ring3` / `testpoint:hole` 折叠后都是 `testpoint`，
        所以它们**不算歧义**，直接取分最高的那个（细分标签保留在 `chosen` 里，
        作证据）；只有折叠后仍然不同（比如 `testpoint` vs `pin`）才留给 LLM。
        """
        n = {"unanimous": 0, "folded": 0}
        for s in self.slots.values():
            if s.resolved() or not s.alts:
                continue
            if len({a.key() for a in s.alts}) == 1:
                s.chosen = s.alts[0].value
                s.decided_by = "rule:unanimous"
                n["unanimous"] += 1
            elif len({self.fold_value(a.value) for a in s.alts}) == 1:
                s.chosen = s.alts[0].value
                s.decided_by = "rule:fold"
                n["folded"] += 1
        return n

    # ---- 读 ----
    def unresolved(self, margin: float = MARGIN) -> list:
        """还没定、且**折叠后仍然不同**、需要外部裁决的槽位（最难的排前面）。"""
        out = []
        for s in self.slots.values():
            if s.resolved() or not s.alts:
                continue
            if len({self.fold_value(a.value) for a in s.alts}) <= 1:
                continue
            out.append(s)
        return sorted(out, key=lambda s: (s.best().score if s.best() else 0.0))

    def decided(self) -> list:
        return [s for s in self.slots.values() if s.resolved()]

    def collapse_types(self) -> dict:
        """槽位 -> 官方类名。`xxx.type` 型槽位按 folds 折叠。"""
        out = {}
        for sid, s in self.slots.items():
            if not sid.endswith(".type"):
                continue
            v = s.chosen if s.resolved() else (s.best().value if s.best() else None)
            out[sid] = self.folds.get(v, v)
        return out

    def to_dict(self) -> dict:
        return {"case": self.case, "folds": self.folds,
                "slots": {k: v.to_dict() for k, v in self.slots.items()}}

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    def decision_log(self) -> list:
        """给 `decision_log.json` 用的紧凑记录。"""
        rows = []
        for sid, s in self.slots.items():
            rows.append({
                "slot": sid,
                "question": s.question,
                "n_alts": len(s.alts),
                "chosen": s.chosen,
                "decided_by": s.decided_by or "PENDING",
                "alts": [{"value": a.value, "score": round(a.score, 4), "source": a.source}
                         for a in s.alts],
            })
        return rows

    @staticmethod
    def load(path) -> "CaseHypotheses":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        h = CaseHypotheses(d.get("case", ""), folds=d.get("folds", {}))
        for sid, sd in (d.get("slots") or {}).items():
            s = Slot(slot_id=sid, question=sd.get("question", ""),
                     context=sd.get("context", {}), chosen=sd.get("chosen"),
                     decided_by=sd.get("decided_by"))
            s.alts = [Alt(**a) for a in sd.get("alts", [])]
            h.slots[sid] = s
        return h


# ------------------------------------------------------------------ 自检
def _demo():
    sys.stdout.reconfigure(encoding="utf-8")
    h = CaseHypotheses("0050")
    h.fold("ring3", "testpoint"); h.fold("hole", "testpoint"); h.fold("dot", "testpoint")
    h.add("t0007.type", "ring3", 0.85, "cv:ring3", {"n_rings": 3, "r_max": 20.1},
          "t0007 是什么元件？", {"cx": 2996, "cy": 212})
    h.add("t0007.type", "hole", 0.80, "cv:hole", {"r": 5.0, "circ": 0.90})
    h.add("t0021.type", "testpoint", 0.85, "cv:ring3", {"n_rings": 3})
    h.add("t0021.type", "testpoint", 0.30, "cv:dot", {"r": 6.0})
    h.add("t0021.key", "TP21", 0.71, "ocr", {"text": "TP21", "cer": 0.0})
    h.add("t0021.key", "TP2", 0.55, "ocr", {"text": "TP2", "cer": 0.25})
    h.add("t0021.key", "TP27", 0.40, "rule:sequence", {"n_pins": 20})
    print("auto_resolve:", h.auto_resolve())
    print("\n未决槽位（要交给 LLM）:")
    for s in h.unresolved():
        print(f"  {s.slot_id}  {s.question}  候选 " +
              " | ".join(f"{a.value}({a.score:.2f} {a.source})" for a in s.alts))
    print("\n已决:", [(s.slot_id, s.chosen, s.decided_by) for s in h.decided()])
    print("\n折叠后的类名:", h.collapse_types())
    # ★ 账本自检（第四项「不删除只削弱」）：改写要留 history + 记事件 + 降权分 > 0。
    h.resolve("t0021.key", "TP21", by="rule:seq")
    h.resolve("t0021.key", "TP2", by="llm:demo")
    hist = h.slots["t0021.key"].history
    assert hist and hist[-1]["value"] == "TP21", hist
    st = LEDGER.stats()
    assert st["n_demotions"] >= 1 and st["n_events"] >= 1, st
    assert st["strictly_positive"], "降权分必须 > 0（0 在很多下游等于删除）"
    print("账本自检 ✓", st)
    print("  事件:", [(e.kind, e.where) for e in LEDGER.events])
    print("  重扫队列:", LEDGER.touched_slots())
    out = Path(r"Y:\workspace\EDA\work\preview\hypo_demo.json")
    h.save(out)
    print(f"\n写到 {out}  ({out.stat().st_size} B)")
    h2 = CaseHypotheses.load(out)
    assert h2.collapse_types() == h.collapse_types()
    print("load 回来一致 ✓")


if __name__ == "__main__":
    _demo()
