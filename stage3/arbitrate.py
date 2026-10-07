# -*- coding: utf-8 -*-
"""第三阶段 · 仲裁：把第二阶段的歧义槽位交给 LLM（或本地 stub）做**选择题**。

契约（沿用 `stage1/llm.py`）：**只做多选题，不让模型自由生成** —— 官方按 key 匹配
元件，编一个值就是多一个错元件。模型只能从 stage2 已经读出来的候选里挑一个。

本机**没有** `LLM_BASE_URL` / `LLM_MODEL`（`Get-ChildItem env:` 实测无匹配），
所以 `mode="auto"` 会落到 **stub**。stub 不是假动作，它做的是本地也能做的逻辑：
**全局唯一性** —— 同一张图里位号唯一，撞号的低分者让位给次优候选。
这治的正是 `stage2` 逐元件独立取最优时必然出现的撞号（两元件都读成 `R7`）。

`mode="llm"` 时把 `hy.unresolved()` 的槽位交给 `stage1.llm.arbitrate`（线程池、只接受
落在候选集合里的答案）。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cvparts.hypo import MARGIN, CaseHypotheses          # noqa: E402

DESIG_SLOT = "desig:%s"


def build(case_id: str, assigned: dict, components=None) -> CaseHypotheses:
    """把 stage2 的 `desig_slots` 变成 `CaseHypotheses`。

    每个候选读法是一个 `Alt`；`auto_resolve()` 先把只有一个候选的槽位按
    `rule:unanimous` 定掉，剩下的才是真歧义。
    """
    ctype = {c["id"]: c.get("type") for c in (components or [])}
    hy = CaseHypotheses(case=case_id)
    for cid, cands in (assigned.get("desig_slots") or {}).items():
        for c in cands:
            if not c.get("canon"):
                continue
            hy.add(DESIG_SLOT % cid, c["canon"],
                   score=float(c.get("score") or 0.0), source="ocr",
                   evidence={"文本": c.get("text"), "置信度": c.get("conf"),
                             "框": c.get("box"), "离框": c.get("d")},
                   question="元件 %s（类型 %s，框 %s）的位号是哪一个？"
                            % (cid, ctype.get(cid), c.get("box")),
                   context={"component": cid, "type": ctype.get(cid)})
    hy.auto_resolve()
    return hy


def _stub_pick(slot, taken):
    """本地仲裁：优先挑一个**没被更自信的元件占掉**的候选。"""
    for a in sorted(slot.alts, key=lambda x: -x.score):
        if a.value not in taken:
            return a.value, "stub:unique"
    b = slot.best()
    return b.value, "stub:fallback"


def arbitrate(hy: CaseHypotheses, mode: str = "auto", workers: int = 8,
              timeout: Optional[float] = None, log=print,
              batch: int = 0, max_slots: int = 0) -> dict:
    """`mode`：auto / stub / llm / off。

    ⚠️ **`auto` 现在落到 `off`**（2025-10-04 实测改的，见下）。想真的走 LLM 必须
    显式写 `mode="llm"`，并且设 `STAGE3_AUTO_LLM=1` 才会让 `auto` 自动启用。

    为什么要这么保守 —— 两条实测结论：

    1. **stub 是错的默认**：stub 会拿「槽位里最自信的候选」对所有槽位做一次全局重排，
       而 `assign.py` 已经做过同一件事且做得更好。220 张实测
       `work\\stage2_v5` 9.8533 → `work\\stage3_v5`(stub) **9.6030**
       （component tp 858→843、extra 1252→**1713**）。没 LLM 时第三阶段就该什么都不做。

    2. **LLM 臂在官方口径上是 0 收益，而且贵 10 小时**。8 张 A/B
       （`work\\tools\\diag\\_llmab.py`，`work\\stage2_v14`）：
       `mode=off` 与 `mode=llm` 的**每一项指标逐位相同**（总分都是 18.3529，
       comp F1 0.4749 / tp 185 / fp 171 / fn 174、pin 0.0041、netH 0.1119、line 0.0088），
       逐 case diff **改动字段 0 个**；而耗时 `off` ≈0 s vs `llm` **1293.4 s（161.7 s/张）**
       ⇒ 220 张要 ~10 小时，直接顶穿 2 小时预算。

       根因是**结构性的、不是接线 bug**：`_slot_block`（`stage1/llm.py:170`）把每个候选的
       **置信度/分数**都写在提示里，而 `qwen3.5-9b` 是**纯文本模型、看不到图**。
       它拿到的唯一有序信号就是那个分数，于是必然选「分数最高的候选」——
       而那正是 `assign.py` 已经选中的那个 ⇒ 输出逐字节不变。
       要让第三阶段真正增值，模型必须**看到局部图像**（VLM，或至少把切图送进去），
       否则它只是在重述 stage2 的 argmax。文本模型再大也补不上这个信息缺口。
    """
    slots = hy.unresolved(margin=MARGIN)
    if mode == "auto":
        allow_auto = os.environ.get("STAGE3_AUTO_LLM", "0") not in ("0", "false", "False", "")
        if not allow_auto:
            mode = "off"
        else:
            try:
                from stage1 import llm as llm_mod
                mode = "llm" if llm_mod.available() else "off"
            except Exception:
                mode = "off"
    t0 = time.time()

    if mode == "off" or not slots:
        return {"mode": mode, "n_slots": 0, "n_decided": 0,
                "seconds": round(time.time() - t0, 3)}

    if mode == "llm":
        from stage1 import llm as llm_mod
        kw = {"workers": workers, "log": log}
        if timeout is not None:
            kw["timeout"] = timeout
        if batch:
            kw["batch"] = batch
        if max_slots:
            kw["max_slots"] = max_slots
        r = llm_mod.arbitrate(hy, slots=slots, **kw)
        return {"mode": "llm", "n_slots": len(slots),
                "n_decided": r.get("resolved", 0),
                "seconds": round(time.time() - t0, 2), "llm": r}

    # ---- stub：最自信的槽位先占坑，撞号的让位 ----
    taken = {s.chosen for s in hy.slots.values()
             if s.resolved() and isinstance(s.chosen, str)}
    decided: Dict[str, str] = {}
    for s in sorted(slots, key=lambda s: -max(a.score for a in s.alts)):
        val, why = _stub_pick(s, taken)
        taken.add(val)
        hy.resolve(s.slot_id, val, by=why)
        decided[s.slot_id] = val
    return {"mode": "stub", "n_slots": len(slots), "n_decided": len(decided),
            "seconds": round(time.time() - t0, 3), "decided": decided}


def apply(assigned: dict, hy: CaseHypotheses) -> dict:
    """把仲裁结果写回 `assigned`（就地改 `designator.canon`），供 `build_doc` 重装配。

    只改 `canon`，原始读法 `text` 与置信度保留为证据，`decision_log.json` 里能追溯。
    """
    for cid, rec in (assigned.get("comps") or {}).items():
        s = hy.slots.get(DESIG_SLOT % cid)
        if s is None or not s.resolved() or not isinstance(s.chosen, str):
            continue
        old = rec.get("designator") or {}
        rec["designator"] = dict(old, canon=s.chosen, by=s.decided_by)
    return assigned


def dedup(assigned: dict, log=print, ledger: list | None = None) -> int:
    """重装配前的最后一道：同名位号**只能有一个主人**（业主 m25181）。

    官方判分按**位号建键**（`metrics.assign_components` 用 `set(gc) & set(pc)`），
    所以提交文档里同一个位号出现两次是**非法**的 —— 这就是「相邻的两个框争一个
    所有权，最终只会归属于某一个」在位号层的样子。

    与旧版的差别（旧版只有三行：分高的留、分低的 `designator = None`）：
    * 走 `cvparts.own.thin` —— 与 stage1 的框归属**同一个实现**，争抢的定义统一为
      「claim 同一个 key」（`frac=0, floor_px=inf` ⇒ 同类必争，与距离无关）；
    * **判负者的主张不销毁** —— 它拿到的不再是一个光秃秃的 `None`，而是
      `designator_dropped = {was, conf, lost_to, dev}`，账同时进 `ledger`。
    """
    from cvparts import own as own_mod

    claimed = []
    for cid, rec in (assigned.get("comps") or {}).items():
        d = rec.get("designator")
        if not d or not d.get("canon"):
            continue
        claimed.append((cid, rec, str(d["canon"]), float(d.get("conf") or 0.0)))

    # `STAGE3_OWN=off` 回到旧行为（静默置 None），用于 A/B 对照
    _on = (os.environ.get("STAGE3_OWN", "on") or "on").strip().lower()
    if _on in ("off", "none", "0", ""):
        by_old: Dict[str, tuple] = {}
        for cid, rec, key, conf in claimed:
            if key not in by_old or conf > by_old[key][0]:
                if key in by_old:
                    by_old[key][1]["designator"] = None
                by_old[key] = (conf, rec)
            else:
                rec["designator"] = None
        return len(by_old)

    led: list = []
    parties = [own_mod.Party(id=cid, box=tuple(rec.get("bbox") or (0, 0, 0, 0)),
                             kind=key, score=conf, meta={"cid": cid})
               for cid, rec, key, conf in claimed]
    kept = own_mod.thin(parties, frac=0.0, floor_px=1e9, same_kind=True,
                        rule="same-designator", ledger=led)
    if ledger is not None:
        ledger.extend(led)

    keep_ids = {p.id for p in kept}
    by: Dict[str, tuple] = {}
    for cid, rec, key, conf in claimed:
        if cid in keep_ids:
            by[key] = (conf, rec)
            continue
        rec["designator"] = None
        # ★ 不销毁：记下它本来主张什么、多可信、输给谁
        loser = next((e for e in led if e.get("dropped") == cid), None)
        rec["designator_dropped"] = {
            "was": key, "conf": round(conf, 4),
            "lost_to": (loser or {}).get("kept"),
            "dev": (loser or {}).get("dev"),
        }
    if led:
        log(f"  独家归属：{len(led)} 个位号主张被改判（见 designator_dropped）")
    return len(by)
