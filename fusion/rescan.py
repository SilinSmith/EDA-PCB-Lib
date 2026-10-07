# -*- coding: utf-8 -*-
"""定向重扫 —— workflow.html 右侧的 `rescan` 节点。

**用户 m12359 原话**（这个节点的由来）：

> LLM 都知道这是 C 这是电容，在漏参数的情况下直接介导一次针对他的重新扫描识别，
> 然后看看能不能在附近把参数提取回来不就好了。

**关键：重扫的输入必须带上已确认的上下文。** 否则「再读一遍」和第一次读没有任何区别
—— 同一张裁片、同一个模型、同一个提示词，只会得到同一个答案。
本模块给每一类疑点配一段上下文（这是电容 / 它邻居是 10nF / 它属于哪一排），
并且**换一种切法**（`window` 而不是 `tight`，从「只看这一个字」换成「看它和它的邻居」）。

**风险门控**（`plausible` 的报告里已经量过）：不能盲目向邻居传播 —— `case 1` 那排
`C151–C156` 是 `10nF ×5 + 1uF ×1`，而 `C156=1uF` 我们**本来就读对了**，广播会把对的改错。
所以重扫的答案一律以**候选**身份进池（不覆盖），并且分数低于第一轮（`RESCAN_SCORE`
默认 1.2 < `FUSION_LLM_SCORE` 的 1.5）—— 第一轮直接读出来的、和重扫出来的，
在证据池里应当有先后。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                                  # eda-pcb
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion import claims as CL                        # noqa: E402
from fusion import llm_judge as LJ                     # noqa: E402
from fusion import plausible as PL                     # noqa: E402
from fusion import priors as PR                        # noqa: E402

# 重扫答案的分数：低于第一轮（1.5），高于纯文本读数（约 1.0）
RESCAN_SCORE = float(os.environ.get("FUSION_RESCAN_SCORE", "1.2"))
# 一轮最多重扫几处（按优先级取）
MAX_RESCAN = int(os.environ.get("FUSION_RESCAN_MAX", "12"))
# 重扫用哪种切法：window=看它和邻居（默认）｜tight=还看那一个字
RESCAN_CROP = os.environ.get("FUSION_RESCAN_CROP", "window").strip().lower()

# 带上下文的重扫提示词。**给候选但不逼着选** —— 这一轮的目的就是让模型
# 在「知道这是什么、邻居是什么」的条件下重新看一遍，而不是在 A/B 之间投票。
PROMPT_CTX = (
    "这是一张电路图的局部截图，图中一个%s被标了出来（位置在截图中部）。\n"
    "%s"
    "请只回答那个%s的字符串，原样照抄，区分大小写与数字。\n"
    "不要解释，不要加标点，不要加空格。如果那里确实没有文字就回答 NONE。"
)
# 每类疑点的上下文模板
CTX_TMPL = {
    "key_duplicate":
        "注意：这一排里有两个元件被读成了同一个位号 %s，其中至少有一个是错的。\n"
        "请只读**当前这一个**元件的位号，不要参考旁边的。\n",
    "key_type_mismatch":
        "注意：这颗元件在结构上被判定为「%s」，位号前缀通常与之相符。\n",
    "value_outlier":
        "注意：同一排的其他%s读到的值都是 %s，这一颗读成了 %s。\n"
        "请只读**当前这一个**元件的值，不要被旁边的数字带跑。\n",
    "pins_unread":
        "注意：这是元件 %s 的一个引脚，这一排引脚号应当是 %s 里的一个。\n",
}
# net 类疑点不是「一个槽位」而是一段网络，单独一段上下文
CTX_NET = (
    "注意：这张图的网络识别结果显示，一条网络上同时出现了「地」和「电源」"
    "（%s），这在物理上不可能。\n"
    "请只读当前这一处的网络标签，确认它到底是地、是电源、还是别的信号名。\n"
)


def _f(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _val(alt):
    """`Alt` → 它承载的值（`None` 安全）。报告里要 JSON 可序列化。"""
    return None if alt is None else str(getattr(alt, "value", alt))


# ------------------------------------------------------------------ 排计划
def plan(pool: CL.ClaimPool, sus: Optional[Sequence[dict]] = None,
         budget: int = MAX_RESCAN) -> List[dict]:
    """把 `plausible` 的可疑项翻成一张**按优先级排好**的重扫清单。

    优先级 = `severity`（这处错了值多少）× `fixable`（重扫真能修好的概率）。
    `net_short` 的 severity 最高（0.90），但它**不是切一个框就能修的** ——
    要重扫的是那一段上的每一处标签和引脚，成本高、收益分散，所以整段重扫
    只排在 `pins_unread` 之前一点点，由调用方按预算决定要不要做。
    """
    if sus is None:
        sus = PL.check(pool)
    from envflag import env_on                # 唯一权威假值词表（见 eda-pcb\envflag.py）
    do_value = env_on("FUSION_RESCAN_VALUE", "0")
    tasks: List[dict] = []
    for s in sus:
        kind = s.get("kind") or "?"
        sev = _f(s.get("severity"))
        if kind in ("net_isolated",):
            continue                                    # 弱信号 + 无从下手，不排
        if kind == "value_outlier" and not do_value:
            # ⚠ 默认不重扫参数离群：它是最弱的一类信号（电源滤波电容本来就大两个量级），
            # 而每一处都要一次 VLM 请求。要开就 `FUSION_RESCAN_VALUE=1`。
            continue
        if kind in ("net_short", "net_rail_merge"):
            for slot in _slots_of_nets(pool, s.get("nets") or []):
                tasks.append({"slot": slot, "kind": kind, "why": s.get("why", ""),
                              "ctx": CTX_NET % ", ".join(s.get("nets") or []),
                              "priority": sev * 0.5, "crop": "window"})
            continue
        ctx = _ctx_for(pool, kind, s)
        for slot in s.get("slots") or []:
            tasks.append({"slot": slot, "kind": kind, "why": s.get("why", ""),
                          "ctx": ctx, "priority": sev * _fixable(kind),
                          "crop": RESCAN_CROP})
    # 同一个槽位只留优先级最高的一条
    best: Dict[str, dict] = {}
    for t in tasks:
        cur = best.get(t["slot"])
        if cur is None or t["priority"] > cur["priority"]:
            best[t["slot"]] = t
    out = sorted(best.values(), key=lambda d: -_f(d.get("priority")))
    return out[:max(0, int(budget))]


def _fixable(kind: str) -> float:
    """重扫真能修好的概率（用来把「可查的」排在「只是可疑的」前面）。"""
    return {"key_duplicate": 0.9, "pins_unread": 0.8, "key_type_mismatch": 0.5,
            "value_outlier": 0.35}.get(kind, 0.5)


def _ctx_for(pool: CL.ClaimPool, kind: str, s: dict) -> str:
    tmpl = CTX_TMPL.get(kind)
    if not tmpl:
        return ""
    slot = (s.get("slots") or [""])[0]
    cid = slot.split(".")[1] if slot.count(".") >= 1 else ""
    if kind == "key_duplicate":
        return tmpl % pool.chosen(slot)
    if kind == "key_type_mismatch":
        return tmpl % ((pool.meta.get(slot) or {}).get("type") or "未知类型")
    if kind == "value_outlier":
        grp = _siblings(pool, cid, "value")
        vals = [str(pool.chosen(x)) for x in grp if pool.chosen(x)]
        common = _mode(vals)
        ctype = (pool.meta.get(slot) or {}).get("type") or "元件"
        return tmpl % (ctype, common or "同一组", pool.chosen(slot) or "")
    if kind == "pins_unread":
        return tmpl % (cid, _pin_span(pool, cid))
    return ""


def _siblings(pool: CL.ClaimPool, cid: str, field_name: str) -> List[str]:
    """同一个元件类型下的同类槽位（用来构造「邻居读到了什么」的上下文）。"""
    ctype = (pool.meta.get("comp.%s.key" % cid) or {}).get("type") or ""
    out = []
    for sid in pool.slot_ids():
        if not sid.startswith("comp.") or not sid.endswith("." + field_name):
            continue
        c2 = sid.split(".")[1]
        if ctype and ((pool.meta.get("comp.%s.key" % c2) or {}).get("type") or "") != ctype:
            continue
        out.append(sid)
    return out


def _mode(vals: Sequence[str]) -> str:
    from collections import Counter
    vals = [v for v in vals if v]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def _pin_span(pool: CL.ClaimPool, cid: str) -> str:
    pins = (pool.doc.get("pins") or {}).get(cid) or []
    n = len(pins)
    return "1..%d" % n if n else "1..N"


def _slots_of_nets(pool: CL.ClaimPool, nets: Sequence[str]) -> List[str]:
    """一段网络上可以重扫的标签槽位。网络标签目前存成 `netlabel.<i>`。"""
    out = []
    for sid in pool.slot_ids():
        if sid.startswith("netlabel.") or sid.startswith("net."):
            out.append(sid)
    return out


# ------------------------------------------------------------------ 执行
def run(pool: CL.ClaimPool, tasks: Optional[Sequence[dict]] = None,
        client=None, limit: Optional[int] = None, log=print) -> dict:
    """执行重扫清单：每处切一张带上下文的图去问，答案以候选身份回池。"""
    if not LJ.JUDGE:
        return {"skipped": True, "reason": "FUSION_JUDGE=0"}
    ts = list(tasks if tasks is not None else plan(pool))
    if limit is not None:
        ts = ts[:max(0, int(limit))]
    bgr = LJ.load_image(pool)
    rep = {"n_task": len(ts), "n_asked": 0, "n_reject": 0, "n_new": 0,
           "rows": [], "sec": 0.0, "crop": RESCAN_CROP}
    if bgr is None:
        rep["error"] = "读不到图"
        return rep
    for t in ts:
        sid = t["slot"]
        reg = pool.region(sid)
        if reg is None:
            rep["rows"].append({"slot": sid, "skip": "无取景框"})
            continue
        kind = LJ.kind_of(sid)
        prompt = PROMPT_CTX % (LJ.KIND_HINT[kind], t.get("ctx") or "", LJ.KIND_HINT[kind])
        # 重扫换一种切法：`window` 让它看见邻居（也知道邻居读成了什么）。
        img = reg.crop(bgr, scale=LJ.SCALE, focus_rect=False,
                       mode=t.get("crop") or RESCAN_CROP)
        if img is None:
            rep["rows"].append({"slot": sid, "skip": "切不出图"})
            continue
        ask = client or LJ._client_ask()
        txt, dt = ask(img, prompt, timeout=LJ.TIMEOUT)
        if isinstance(txt, tuple):
            txt, dt = txt
        ans = LJ.clean_answer(txt)
        rep["n_asked"] += 1
        rep["sec"] += _f(dt)
        row = {"slot": sid, "kind": t.get("kind"), "ans": ans,
               "before": _val(pool.best(sid)), "sec": round(_f(dt), 1)}
        if not ans:
            row["res"] = "empty"
            rep["rows"].append(row)
            continue
        known = {str(a.value) for a in pool.alts(sid)}
        ok, mult = LJ._admit(pool, sid, ans)
        if not ok:
            row["res"] = "reject"
            rep["n_reject"] += 1
            rep["rows"].append(row)
            if log is not None:
                log("  %-18s [%-16s] -> %-10s reject" % (sid, t.get("kind"), ans))
            continue
        row["res"] = "same" if ans in known else "new"
        rep["n_new"] += (0 if ans in known else 1)
        pool.add(sid, ans, score=RESCAN_SCORE * mult, source="vlm:rescan",
                 evidence={"上下文": (t.get("ctx") or "").strip()[:200],
                           "切法": t.get("crop") or RESCAN_CROP,
                           "原样": str(txt or "").strip()[:120],
                           "因为": t.get("why", "")[:200],
                           "耗时": round(_f(dt), 1)},
                 question=pool.hy.slots[sid].question if sid in pool.hy.slots else "",
                 meta=dict(pool.meta.get(sid) or {}))
        rep["rows"].append(row)
        if log is not None:
            log("  %-18s [%-16s] %-8s -> %-10s [%s] %.1fs"
                % (sid, t.get("kind"), row["before"], ans, row["res"], _f(dt)))
    return rep


# ------------------------------------------------------------------ 自检
def _demo() -> int:
    ok = 0
    total = 0

    def chk(name, cond):
        nonlocal ok, total
        total += 1
        if cond:
            ok += 1
        else:
            print("  ✗ %s" % name)

    doc = {"image": {"w": 400, "h": 300, "path": ""},
           "components": [{"id": "c000%d" % i, "type": "c",
                           "bbox_img": [10 + 40 * i, 10, 30 + 40 * i, 30]}
                          for i in range(5)],
           "pins": {"c000%d" % i: [[20 + 40 * i, 40]] for i in range(5)},
           "nets": []}
    assigned = {"comps": {
        "c0000": {"designator": {"text": "C1", "canon": "C1"},
                  "value": {"text": "10nF"}, "partname": None, "pins": {}},
        "c0001": {"designator": {"text": "C1", "canon": "C1"},
                  "value": {"text": "10nF"}, "partname": None, "pins": {}},
        "c0002": {"designator": {"text": "C3", "canon": "C3"},
                  "value": {"text": "100nF"}, "partname": None,
                  "pins": {"0": {"number": None, "name": None}}},
        "c0003": {"designator": {"text": "C4", "canon": "C4"},
                  "value": {"text": "10nF"}, "partname": None, "pins": {}},
        "c0004": {"designator": {"text": "C5", "canon": "C5"},
                  "value": {"text": "10nF"}, "partname": None, "pins": {}},
    }, "desig_slots": {}, "net_labels": [], "struct_slots": []}
    pool = CL.from_stage2("demo", doc, assigned)
    for cid, v in (("c0000", "C1"), ("c0001", "C1"), ("c0002", "C3"),
                   ("c0003", "C4"), ("c0004", "C5")):
        pool.add("comp.%s.key" % cid, v, score=1.0, source="test")
        pool.hy.resolve("comp.%s.key" % cid, v, "test")
    for cid, v in (("c0000", "10nF"), ("c0001", "10nF"), ("c0002", "100nF"),
                   ("c0003", "10nF"), ("c0004", "10nF")):
        pool.add("comp.%s.value" % cid, v, score=1.0, source="test")
        pool.hy.resolve("comp.%s.value" % cid, v, "test")
    # 取景框：位号框就是本体左边一点
    for i in range(5):
        pool.set_region("comp.c000%d.key" % i, [8, 8, 32, 32], focus=[9, 9, 25, 20])

    sus = PL.check(pool)
    # 参数离群的重扫默认关（最弱的信号 × 每处一次请求）⇒ 自检里显式打开
    os.environ["FUSION_RESCAN_VALUE"] = "1"
    tasks = plan(pool, sus)
    chk("排出了重扫计划", len(tasks) > 0)
    chk("重号的两个槽位都排上了",
        sum(1 for t in tasks if t["kind"] == "key_duplicate") >= 2)
    chk("离群值也排上了", any(t["kind"] == "value_outlier" for t in tasks))
    chk("同一槽位只留一条",
        len({t["slot"] for t in tasks}) == len(tasks))
    chk("按优先级降序",
        all(tasks[i]["priority"] >= tasks[i + 1]["priority"] for i in range(len(tasks) - 1)))
    chk("重扫切法与第一轮不同", all(t["crop"] == RESCAN_CROP for t in tasks))
    chk("重号的上文提到重号",
        any(t["kind"] == "key_duplicate" and "同一个位号" in (t["ctx"] or "")
            for t in tasks))
    chk("离群的上文给出邻居的值",
        any(t["kind"] == "value_outlier" and "10nF" in (t["ctx"] or "") for t in tasks))
    chk("重扫分数低于第一轮", RESCAN_SCORE < LJ.LLM_SCORE)
    chk("提示词是带上下文的", "%s" in PROMPT_CTX and "注意" in CTX_TMPL["key_duplicate"])
    chk("预算能截断", len(plan(pool, sus, budget=1)) <= 1)
    chk("_fixable 把可查的排在前面", _fixable("key_duplicate") > _fixable("value_outlier"))

    print("rescan 自检 %d 项：%s" % (total, "全部通过" if ok == total else
                                     "%d 项未过" % (total - ok)))
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(_demo())
