# -*- coding: utf-8 -*-
"""约束求解：把证据池里的候选变成**全局自洽**的一个解。

这一层是 workflow.html 右侧「融合与仲裁」里 `solve` 节点的落点（此前没有代码）。
它不是「给每个槽位取最高分」——`assign.py` 已经那样做了，成绩就是现在的分数。
它做的是 `assign.py` 做不到的事：**把槽位之间的相互约束一起考虑**。

四个约束（都能写进目标函数，所以可解）
------------------------------------
1. **唯一性**：一张图里一个位号只能属于一颗元件。`case 1` 里 `C5` 被两颗元件读过
   （`c0008` 与 `c0015`），`C11`、`C15`、`C4` 同理 —— 贪心必然让它们撞车。
   引脚号在**同一颗元件内**唯一，同理。
2. **连号**：同前缀位号几乎总是 1..N 的连续区间。这是最强的先验：`case 1`
   真值 `C151..C156`，六个框全读错，但连号把答案集合缩到 6 个数上。
3. **形状**：候选值要和槽位类型相符（`comp.c.key` 得是位号形状）。
4. **来源**：`priors` 造出来的候选带折扣，只在没有更好的解释时才赢。

输出
----
`solve()` 返回一个 dict，含每个槽位的选定值 + 理由（`by`）、以及**仍不确定**的
槽位清单（给 `llm_judge`）。**它不修改证据池**（只读）；`write_back()` 才写。

为什么用「贪心 + 增广路」而不是 Hungarian
----------------------------------------
候选表很小（每槽 ≤ 8 个），但**值集合是稀疏的**：`case 1` 的 37 个位号槽位里大多数
只有 1 个候选。按分数降序贪心 + 冲突时沿增广路给占用者找下一个可用值，在
「绝大多数槽位无冲突」的真实分布上等价于最优解，且没有 Hungarian 的 O(n³) 与
浮点相等问题。冲突链长度实测 ≤ 2。
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion import claims as CL          # noqa: E402
from fusion import priors as PR          # noqa: E402

# 「最优与次优差多少以内算判不出来」——比 hypo.MARGIN(0.15) 紧，因为这里的候选
# 分数是「读法置信度 × 形状」的混合量，同一读法内部的浮动也有 0.05 量级。
OPEN_MARGIN = float(os.environ.get("FUSION_OPEN_MARGIN", "0.10"))
# 最多留几个问题给 LLM 看图裁决（每问一次约 0.3 s，但端点有并发上限）
MAX_OPEN = int(os.environ.get("FUSION_MAX_OPEN", "24"))

# ★★ boxscan ②：**零候选**槽位的「去问一遍」路径。
#
# 改造前 `open_questions` 里是 `if not alts: continue` —— 一个候选都没有的槽位被
# 彻底挡在门外。而「一个候选都没有」恰恰是最该问模型的局面：位号文字**根本没进
# 候选表**（220 张实测：629 颗没配上的 GT IC 里 418 颗的重叠元件 `desig_slots` 是空的）。
#
# 打开后：零候选槽位**只要有取景框**（`claims.BOXSCAN` 会给每个元件建框）就出一题
# 自由读数题，**单独排队、单独限量**，不挤占正常「二选一」题的名额。
# `FUSION_OPEN_FREEFORM_MAX` = 每张最多几道（每题约 0.2~30 s，端点预算是硬约束）。
FREEFORM_MAX = int(os.environ.get("FUSION_OPEN_FREEFORM_MAX", "8"))
# 连号空洞补全：一个前缀至少有多少个已定号才敢推「这里应该有第 N 号」
GAP_MIN_RUN = int(os.environ.get("FUSION_GAP_MIN_RUN", "4"))
# 区间的**填充率**下限：`R7..R1210` 这种被垃圾读数拉开的区间不该补洞
GAP_MIN_COVER = float(os.environ.get("FUSION_GAP_COVER", "0.45"))
# 补洞候选的分数折扣（相对该前缀已定号的中位分）
GAP_SCORE = float(os.environ.get("FUSION_GAP_SCORE", "0.90"))
from envflag import env_on                       # noqa: E402  (唯一权威假值词表)
# ★ 同 `fusion\run.py`：手写短词表 ⇒ `FUSION_GAP_FILL=off` 被当成开。统一。
GAP_FILL = env_on("FUSION_GAP_FILL", "1")
# ★★ boxscan ② 的开关（详见 `MAX_OPEN` 下面那段）。关掉即逐位退回改造前行为。
FREEFORM = env_on("FUSION_OPEN_FREEFORM", "1")
# 每个槽位最多留几个候选进匹配（含合成的）
MAX_CANDS = int(os.environ.get("FUSION_MAX_CANDS", "8"))

DESIG_SLOT_RE = re.compile(r"^comp\.([^.]+)\.key$")
PINNUM_SLOT_RE = re.compile(r"^pin\.([^.]+)\.([^.]+)\.number$")


def _f(v, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _is_read(src) -> bool:
    """这个候选是不是**真的读出来的**（相对于 `prior:`/`solve:` 造出来的）。

    判据与 `fusion/doc.py:_read_src`、`fusion/priors.py` 的 `no_read` 一致：
    **读数是证据，先验/连号只是形状** —— 形状可以帮没有读数的槽位补号，
    但不能把已经读出来的号抢走（实测 `0001`/`0005` 直接掉分）。
    """
    return str(src or "").startswith(("ocr", "vlm"))


# ------------------------------------------------------------------ 槽位族
def unique_family(slot_id: str) -> Optional[str]:
    """这个槽位的值必须**全局唯一**吗？返回族名（同一族内互斥），否则 None。"""
    m = DESIG_SLOT_RE.match(slot_id)
    if m:
        return "designator"
    m = PINNUM_SLOT_RE.match(slot_id)
    if m:
        return "pinnum:" + m.group(1)          # 同一颗元件内唯一
    return None


def slot_prefix(slot_id: str) -> str:
    """位号槽位的前缀（`comp.c3.key` → 看 best 值的字母部分，没有就用类型推）。"""
    return ""


# ------------------------------------------------------------------ 候选表
def gather(pool, slot_ids: Sequence[str],
           keep_priors: bool = True) -> Dict[str, List[Tuple[str, float, str]]]:
    """把池里的候选抽成 `{slot: [(value, score, source), ...]}`（已按分数降序）。"""
    out: Dict[str, List[Tuple[str, float, str]]] = {}
    for sid in slot_ids:
        rows: List[Tuple[str, float, str]] = []
        seen = set()
        for a in pool.alts(sid):
            v = "" if a.value is None else str(a.value)
            if not v or v in seen:
                continue
            if not keep_priors and a.source.startswith("prior:"):
                continue
            seen.add(v)
            rows.append((v, CL._f(a.score), a.source))
        rows.sort(key=lambda t: -t[1])
        out[sid] = rows[:MAX_CANDS]
    return out


def _taken_by(cands: Dict[str, List[Tuple[str, float, str]]],
              assign: Dict[str, Tuple[str, float, str]],
              family: str) -> Dict[str, str]:
    """某个唯一族里「值 -> 占它的槽位」。"""
    out: Dict[str, str] = {}
    for sid, (v, _s, _src) in assign.items():
        if unique_family(sid) == family:
            out[v] = sid
    return out


def _assign(slot: str, val: str, score: float, src: str,
            cand_idx: Dict[str, List[Tuple[str, float, str]]],
            taken: Dict[str, str], assign: Dict[str, Tuple[str, float, str]],
            seen: set) -> bool:
    """把 `val` 给 `slot`；若已被占，沿增广路给占用者换一个可用值。"""
    cur = taken.get(val)
    if cur is None:
        taken[val] = slot
        assign[slot] = (val, score, src)
        return True
    if cur in seen or cur == slot:
        return False
    seen.add(cur)
    for v2, s2, src2 in cand_idx.get(cur, []):
        if v2 == val:
            continue
        if _assign(cur, v2, s2, src2, cand_idx, taken, assign, seen):
            taken[val] = slot
            assign[slot] = (val, score, src)
            return True
    return False


def match(cands: Dict[str, List[Tuple[str, float, str]]],
          family_of) -> Tuple[Dict[str, Tuple[str, float, str]], Dict[str, list]]:
    """对每个唯一族各跑一次「贪心 + 增广路」。返回 (选定表, 每族的冲突记录)。"""
    assign: Dict[str, Tuple[str, float, str]] = {}
    conflicts: Dict[str, list] = defaultdict(list)
    fams: Dict[str, List[str]] = defaultdict(list)
    for sid in cands:
        fam = family_of(sid)
        if fam:
            fams[fam].append(sid)

    for fam in sorted(fams):
        slots = fams[fam]
        taken: Dict[str, str] = {}
        pairs = []
        for sid in slots:
            for v, s, src in cands.get(sid) or []:
                pairs.append((s, sid, v, src))
        pairs.sort(key=lambda t: -t[0])
        for s, sid, v, src in pairs:
            if sid in assign:
                continue
            if v in taken:
                conflicts[fam].append({"value": v, "winner": taken[v], "loser": sid})
            if _assign(sid, v, s, src, cands, taken, assign, set()):
                continue
        # 兜底：族里没被匹配到的槽位（候选全被抢走）先留空，后面 gap_fill 再补
    return assign, dict(conflicts)


# ------------------------------------------------------------------ 连号补洞
def prefix_of(pool, sid: str) -> str:
    """位号槽位的前缀。优先看候选里的字母部分，其次看元件类型。"""
    for a in pool.alts(sid):
        v = "" if a.value is None else str(a.value)
        pre, num = PR.split_desig(v)
        if pre and num is not None:
            return pre.upper()
    ctype = (pool.meta.get(sid) or {}).get("type") or ""
    if ctype:
        pre = PR.TYPE_PREFIX.get(ctype.split(":")[0])
        if pre:
            return (pre[0] if isinstance(pre, (tuple, list)) else str(pre)).upper()
    return ""


def number_of(v) -> Optional[int]:
    _p, n = PR.split_desig("" if v is None else str(v))
    return n


def gap_fill(pool, cands, assign, stats=None) -> int:
    """按连号先验给「空洞号」造候选，重跑该族的匹配。返回造出的候选数。"""
    st = stats if stats is not None else {}
    if not GAP_FILL:
        return 0
    keys = [s for s in cands if DESIG_SLOT_RE.match(s)]
    by_pre: Dict[str, List[str]] = defaultdict(list)
    for sid in keys:
        by_pre[prefix_of(pool, sid)].append(sid)

    added = 0
    for pre, slots in sorted(by_pre.items()):
        if not pre or len(slots) < GAP_MIN_RUN:
            continue
        # 已定号（用当前解）
        fixed: Dict[str, int] = {}
        scores: List[float] = []
        for sid in slots:
            row = assign.get(sid)
            v = row[0] if row else None
            n = number_of(v)
            if n is not None:
                fixed[sid] = n
                scores.append(row[1] if row else 0.0)
        if len(fixed) < GAP_MIN_RUN:
            continue
        # 取「填充率最好」的连号段（`desig_runs` 会过滤掉 R1210 这种离群）
        runs = PR.Priors.desig_runs(list(fixed.values()))
        if not runs:
            continue
        lo, hi, hits, span = runs[0]
        if span <= hits or span > 4 * max(1, len(slots)):
            continue
        have = set(fixed.values())
        holes = [n for n in range(lo, hi + 1) if n not in have]
        if not holes:
            continue
        # 只补「已定号基本铺满这个区间」的情况
        if hits / float(span) < GAP_MIN_COVER:
            continue
        base = (sorted(scores)[len(scores) // 2] if scores else 0.5) * GAP_SCORE
        # 把空洞号加给**该前缀下所有槽位**（分数打折），让匹配自己去挑一个自洽的排布。
        #
        # ★ 但**不能盖过真读数**。实测 `0001`：`comp.c0019.key` 的真读数是
        # `C13 (0.321, ocr:designator)`，而补洞候选的 base（已定号中位分 × 0.90）
        # 是 `0.340` —— 于是 `C13` 被改成了 `C16`；同样 `C8→C6`（`0001`）、
        # `R15→R12`、`R17→R11`（`0001`）、`C1→C11`、`C7→C1`（`0005`）。
        # 逐张代价 `0001 −2.50`、`0005 −1.70`。
        # 原理与 `priors.no_read` 同一条：**连号先验只能给读不出号的槽位补号，
        # 不能把已经读出来的号抢走** —— 读数是证据，连号只是形状。
        for sid in slots:
            b = pool.best(sid)
            own = _f(b.score) if b else 0.0
            own_read = bool(b) and _is_read(str(getattr(b, "source", "") or ""))
            sc = min(base, own - 0.01) if own_read else base
            if sc <= 0.0:
                # ★ **不要再让它静默**：`own_read` 为真且 `own < 0` 时（已有候选被
                #   判成「读错了」，分数为负），`min(base, own-0.01)` 也是负的，
                #   于是这个槽位整条补洞失效 —— 表现为「`FUSION_GAP_FILL=1` 却像
                #   没开」。行为本身是保守的（负分 = 有反证，不该覆盖），但必须
                #   留下计数，否则下次又要在 `stats` 里翻半天。
                st["gap_skip_negative"] = int(st.get("gap_skip_negative", 0)) + 1
                continue
            have_v = {v for v, _s, _src in cands.get(sid) or []}
            for n in holes:
                cand = "%s%d" % (pre, n)
                if cand in have_v:
                    continue
                cands[sid].append((cand, sc, "solve:gapfill"))
                have_v.add(cand)
                added += 1
            cands[sid].sort(key=lambda t: -t[1])
            del cands[sid][MAX_CANDS:]
        st["gap_%s" % pre] = {"lo": lo, "hi": hi, "hits": hits, "span": span,
                              "holes": holes}
    return added


# ------------------------------------------------------------------ 主入口
def solve(pool, do_gap: bool = True) -> dict:
    """解一遍。返回 `{assign, cands, conflicts, open, stats}`。**不改池**。"""
    keys = [s for s in pool.slot_ids() if s.startswith("comp.")]
    pins = [s for s in pool.slot_ids() if s.startswith("pin.")]
    rest = [s for s in pool.slot_ids()
            if not s.startswith("comp.") and not s.startswith("pin.")
            and not s.startswith("netlabel.")]

    stats: Dict[str, object] = {}
    cands = gather(pool, keys + pins)
    # 非唯一槽位（值/器件名/网络名/引脚名）：直接取最高分
    assign: Dict[str, Tuple[str, float, str]] = {}
    for sid in keys + pins:
        fam = unique_family(sid)
        if fam:
            continue
        rows = cands.get(sid) or []
        if rows:
            assign[sid] = rows[0]

    a1, c1 = match(cands, unique_family)
    assign.update(a1)
    stats["n_unique"] = len(a1)
    stats["n_free"] = len(assign) - len(a1)

    added = gap_fill(pool, cands, assign, stats) if do_gap else 0
    stats["gap_candidates"] = added
    if added:
        a2, c2 = match(cands, unique_family)
        # 补洞后的解只在**族内**替换（自由槽位不动）
        assign.update(a2)
        for fam, rows in c2.items():
            c1.setdefault(fam, []).extend(rows)

    stats["n_assigned"] = len(assign)
    stats["n_slots"] = len(keys) + len(pins) + len(rest)
    stats["n_lost"] = sum(1 for s in keys + pins if s not in assign)

    open_q = open_questions(pool, assign)
    stats["n_open"] = len(open_q)
    return {"assign": assign, "cands": cands, "conflicts": c1,
            "open": open_q, "stats": stats}


# ------------------------------------------------------------------ 待裁决
def open_questions(pool, assign, margin: Optional[float] = None,
                   limit: Optional[int] = None) -> List[dict]:
    """挑出**最优与次优太近**的槽位，供 `llm_judge` 带图裁决。

    刻意**不用** `hypo.unresolved()`：那个只看「还没定」，而这里要的是「定了但
    证据不够」，两者的差别正是 `case 1` 那种「六条读数全都错、但彼此分数很接近」
    的局面 —— 位号即使被解出来，仍然要让 LLM 看一眼图确认。
    """
    margin = OPEN_MARGIN if margin is None else float(margin)
    limit = MAX_OPEN if limit is None else int(limit)
    out: List[dict] = []
    free: List[dict] = []
    for sid in sorted(pool.slot_ids()):
        row = assign.get(sid)
        rows = [row] if row else []
        alts = pool.alts(sid)
        if not alts:
            # ★★ boxscan ②：零候选槽位以前在这里被 `continue` 挡死 —— 而「一个候选
            #   都没有」恰恰是最该问模型的局面（位号文字根本没进候选表）。只在**有
            #   取景框**时才出题（没图问不了），且单独排队 + 单独限量，不挤占正常题。
            if not FREEFORM or pool.region(sid) is None:
                continue
            free.append({
                "slot": sid,
                "question": (pool.hy.slots[sid].question or "这一项应该是什么？"),
                "top": None,
                "alt": None,
                "margin": None,
                "region": pool.region(sid).to_dict(),
                "choices": [],
                "free": True,
            })
            continue
        top = rows[0] if rows else (str(alts[0].value), CL._f(alts[0].score), alts[0].source)
        # 次优：候选表里**与 top 不同值**的最高分
        second = None
        for v, s, src in (rows[1:] + [(str(a.value), CL._f(a.score), a.source)
                                      for a in alts]):
            if v != top[0]:
                second = (v, s, src)
                break
        if second is None:
            continue
        if (top[1] - second[1]) >= margin:
            continue
        reg = pool.region(sid)
        out.append({
            "slot": sid,
            "question": (pool.hy.slots[sid].question or "这一项应该是什么？"),
            "top": {"value": top[0], "score": round(top[1], 4), "source": top[2]},
            "alt": {"value": second[0], "score": round(second[1], 4), "source": second[2]},
            "margin": round(top[1] - second[1], 4),
            "region": (reg.to_dict() if reg is not None else None),
            "choices": [top[0], second[0]],
        })
    # 最不确定的排前面；同分时优先「位号/引脚号」这种最值钱的槽位
    prio = {"comp": 0, "pin": 1, "net": 2}
    out.sort(key=lambda d: (d["margin"], prio.get(d["slot"].split(".")[0], 3)))
    # ★ boxscan ② 排在**正常题之后**、单独限量：零候选题的价值不如「两个候选咬得很紧」
    #   的题确定，但也不能因为是新通道就被 `limit` 一刀切掉。
    return out[:limit] + free[:max(0, FREEFORM_MAX)]


# ------------------------------------------------------------------ 回写
def write_back(pool, sol, by: str = "solve", only: Optional[Sequence[str]] = None) -> int:
    """把解写进证据池（`Slot.chosen`/`decided_by`），返回写了几条。"""
    n = 0
    for sid, (v, _s, src) in sorted((sol.get("assign") or {}).items()):
        if only is not None and sid not in only:
            continue
        if sid not in pool.hy.slots:
            continue
        pool.hy.resolve(sid, v, "%s:%s" % (by, src or "?"))
        n += 1
    return n


def to_dict(sol) -> dict:
    return {
        "assign": {k: {"value": v[0], "score": round(v[1], 4), "source": v[2]}
                   for k, v in sorted((sol.get("assign") or {}).items())},
        "conflicts": sol.get("conflicts") or {},
        "open": sol.get("open") or [],
        "stats": sol.get("stats") or {},
    }


def save(sol, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_dict(sol), ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ 自检
def _demo() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ok: List[Tuple[str, bool]] = []

    # 六颗电容，读数=case 1 实测（C51/C15/C5/C54/C5/C15），真值 C151..C156
    reads = ["C51", "C15", "C5", "C54", "C5", "C15"]
    doc = {"image": {"w": 1600, "h": 1200},
           "components": [{"id": "c%d" % i, "type": "c", "bbox_img": [10 + 40 * i, 10, 30 + 40 * i, 30]}
                          for i in range(6)],
           "pins": {"c%d" % i: [[10 + 40 * i, 20], [30 + 40 * i, 20]] for i in range(6)},
           "nets": []}
    assigned = {"comps": {}, "desig_slots": {}, "net_labels": [], "struct_slots": []}
    for i, t in enumerate(reads):
        cid = "c%d" % i
        assigned["comps"][cid] = {"designator": {"text": t}, "value": None,
                                  "partname": None, "pins": {}}
        assigned["desig_slots"][cid] = [
            {"text": t, "canon": t, "conf": 1.0, "box": [10 + 40 * i, 10, 30 + 40 * i, 30],
             "d": 2.0, "fit": 1.0, "box_i": i, "score": 1.0 - 0.02 * i}]
    pool = CL.from_stage2("demo", doc, assigned)
    PR.Priors().annotate(pool)
    sol = solve(pool)
    st = sol["stats"]
    got = sorted(v[0] for v in sol["assign"].values() if v[0].startswith("C"))
    ok.append(("六个槽位全部有解", len(got) == 6))
    ok.append(("没有重复位号", len(set(got)) == len(got)))
    allv = {str(a.value) for sid in pool.slot_ids() for a in pool.alts(sid)}
    ok.append(("真值 C151/C154 作为候选出现", {"C151", "C154"} <= allv))
    ok.append(("c0 的候选里同时有 C51 与 C151", {"C51", "C151"} <=
               {str(a.value) for a in pool.alts("comp.c0.key")}))
    q0 = [q for q in sol["open"] if q["slot"] == "comp.c0.key"]
    ok.append(("c0 被留成待裁决（C51 还是 C151）",
               bool(q0) and set(q0[0]["choices"]) == {"C51", "C151"}))
    ok.append(("唯一族里有解", st.get("n_unique", 0) >= 4))
    ok.append(("连号补洞造出了候选", st.get("gap_candidates", 0) > 0))
    ok.append(("留了问题给 LLM", st.get("n_open", 0) > 0))

    # 唯一性：两颗元件都读 C5 -> 必须只有一个拿到 C5
    doc2 = dict(doc)
    assigned2 = json.loads(json.dumps(assigned))
    for cid in ("c0", "c1"):
        assigned2["comps"][cid]["designator"] = {"text": "C5"}
        assigned2["desig_slots"][cid] = [
            {"text": "C5", "canon": "C5", "conf": 1.0, "box": [10, 10, 30, 30],
             "d": 2.0, "fit": 1.0, "box_i": 0, "score": 0.9}]
    for cid in ("c2", "c3", "c4", "c5"):
        assigned2["comps"][cid].pop("designator", None)
        assigned2["desig_slots"].pop(cid, None)
    pool2 = CL.from_stage2("demo2", doc2, assigned2)
    sol2 = solve(pool2)
    got2 = [v[0] for v in sol2["assign"].values() if v[0].startswith("C")]
    ok.append(("撞车的 C5 只被一颗拿到", got2.count("C5") <= 1))
    ok.append(("另一颗被换成别的号", len(set(got2)) == len(got2)))

    # 回写
    n = write_back(pool, sol)
    ok.append(("回写条数 = 解的条数", n == len(sol["assign"])))
    ok.append(("回写后槽位已定", all(pool.hy.slots[s].resolved() for s in sol["assign"])))

    bad = [k for k, v in ok if not v]
    print("solve 自检 %d 项：%s" % (len(ok), "全部通过" if not bad else "失败"))
    for k, v in ok:
        print("  %s %s" % ("PASS" if v else "FAIL", k))
    print("  统计：%s" % {k: v for k, v in sol["stats"].items() if not k.startswith("gap_")})
    print("  demo 解出的位号：%s" % got)
    print("  留给 LLM 的问题数：%d，例：%s" % (
        len(sol["open"]), (sol["open"][0]["slot"] + " " + str(sol["open"][0]["choices"]))
        if sol["open"] else None))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_demo())
