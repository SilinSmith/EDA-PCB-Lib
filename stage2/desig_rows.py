# -*- coding: utf-8 -*-
"""行内位号按 x 序一一对应（`STAGE2_DESIG_ROWS`，**默认 0=关**）。

病（`open_GT` case1 的电阻条带，2026-11 取证）：一行竖放电阻，位号印在各自**右侧**：

    GT    1799 R20 / 1848 R19 / 1898 R16 / 1947 R17 / 1999 R18 / 2056 R15
    stage2 1799 R20 / 1848 --- / 1897 R19 / 1947 R16 / 1998 R17 / 2056 R15

「逐颗捡最近文字框」在等距的排里必然往右偏一颗（左邻框总比右邻框近 ~3px），
且 `assign` 按 cid 升序**先到先得**，前面的元件先抢走邻居的位号，最后一个空着。

规则（三条**同时**成立才动手，任何一条不成立就保持现状）：
  1) 簇内元件数 == 簇内可用位号数：位号 = 簇内各元件 `desig_slots` 里
     `^[A-Za-z]{1,3}_?\\d{1,4}$` 的候选，按 `box_i` 去重，同前缀、互不相同，
     且没有一个被**簇外**元件的 `designator.canon` 占着；
  2) 单调：按 x 排序后，各元件「当前位号框（没有就用最近候选框）的 x」非降
     —— 保证我们不是在给一堆乱序/互相穿插的文字框硬配对；
  3) 动作：位号按框 x 排序、元件按中心 x 排序，一对一重配（覆盖现有分配）。
     同时把该元件的 `desig_slots` 收成「只剩这一条」，否则 fusion 的
     `claims.from_stage2` 会按旧候选表全局重解一遍、把结果抹回去。

为什么要动 `desig_slots`：最终键是 `fusion/claims.py` 的全局解写回的，
它只认 `desig_slots`（`assigned[cid]["designator"]` 根本不进池）；只改
`designator` 不改槽位 ⇒ 解出来还是旧的。

离线读数（`work/_rowfix_orc.py`，13 页，`work/_cv_run_rebind0`）：规则只在 5 个簇成立
（case1 ×1、case7 ×2、case10 ×2），其中 4 个簇现状已对（不动），case1 那个 6 元簇
**改对 4 颗（R19/R16/R17/R18 各回到自己那颗）、改错 0**。
"""
from typing import Dict, List, Optional
import re

ROW_TYPES = frozenset(("r", "c", "l"))
ROW_TOL = 15.0          # 行内 y 中心容差
ROW_GAP = 120.0         # 行内相邻元件 x 间距上限（超过就断开成两簇）
ROW_MIN = 3             # 簇内至少这么多元件才动手
DESIG_RE = re.compile(r"^[A-Za-z]{1,3}_?\d{1,4}$")


def _cx(b) -> float:
    return (float(b[0]) + float(b[2])) / 2.0


def _cy(b) -> float:
    return (float(b[1]) + float(b[3])) / 2.0


def _rows(members: List[dict]) -> List[List[dict]]:
    """同 type、y 中心相差 ≤ROW_TOL 聚行，再按 x 邻距 <ROW_GAP 断开。"""
    out: List[List[dict]] = []
    by_type: Dict[str, List[dict]] = {}
    for m in members:
        by_type.setdefault(m["type"], []).append(m)
    for _t, ms in by_type.items():
        rows: List[List[dict]] = []
        for m in sorted(ms, key=lambda d: (d["by"], d["bx"])):
            for row in rows:
                if abs(row[0]["by"] - m["by"]) <= ROW_TOL:
                    row.append(m)
                    break
            else:
                rows.append([m])
        for r in rows:
            r.sort(key=lambda d: d["bx"])
            cur = [r[0]]
            for m in r[1:]:
                if m["bx"] - cur[-1]["bx"] < ROW_GAP:
                    cur.append(m)
                else:
                    if len(cur) >= ROW_MIN:
                        out.append(cur)
                    cur = [m]
            if len(cur) >= ROW_MIN:
                out.append(cur)
    return out


def _pool(cl: List[dict]) -> List[dict]:
    """簇内可用位号槽：按 `box_i` 去重，同一条文字只留距离最近的那份。"""
    pool: Dict[object, dict] = {}
    for m in cl:
        for s in m["slots"]:
            c = str(s.get("canon") or "")
            if not DESIG_RE.match(c) or not s.get("box"):
                continue
            bi = s.get("box_i")
            if bi is None:
                bi = (c, tuple(round(float(x), 1) for x in s["box"][:4]))
            d = float(s.get("d") or 1e9)
            old = pool.get(bi)
            if old is None or d < float(old.get("d") or 1e9):
                pool[bi] = s
    return list(pool.values())


def _cur_x(m: dict) -> float:
    """该元件「现在认的位号框」的 x：有 designator.box 用它，否则用最近候选。"""
    d = m["cur_box"]
    if d:
        return _cx(d)
    best = None
    for s in m["slots"]:
        if s.get("box") and (best is None or float(s.get("d") or 1e9) < float(best.get("d") or 1e9)):
            best = s
    return _cx(best["box"]) if best else m["bx"]


def fix(doc: dict, a: dict) -> dict:
    """返回统计；`a["comps"][cid]` / `a["desig_slots"][cid]` 会被就地改写。"""
    stat: Dict[str, object] = {"on": True, "rows": 0, "changed": 0, "bails": 0, "details": []}
    comps = a.get("comps") or {}
    slots_all = a.get("desig_slots") or {}
    boxes = {}
    for c in doc.get("components") or []:
        b = c.get("bbox_img")
        if b and len(b) >= 4:
            boxes[c.get("id")] = (str(c.get("type") or ""), [float(x) for x in b[:4]])
    members: List[dict] = []
    for cid in comps:
        bt = boxes.get(cid)
        if not bt or bt[0] not in ROW_TYPES:
            continue
        d = comps[cid].get("designator") or {}
        members.append({"cid": cid, "type": bt[0], "bx": _cx(bt[1]), "by": _cy(bt[1]),
                        "cur_box": d.get("box"), "cur": d.get("canon"),
                        "slots": [s for s in (slots_all.get(cid) or [])
                                  if str(s.get("canon") or "")]})
    by_cid = {m["cid"]: m for m in members}
    for cl in _rows(members):
        stat["rows"] += 1
        pool = _pool(cl)
        if len(pool) != len(cl):
            continue
        pres = {re.match(r"^[A-Za-z]{1,3}", str(s["canon"])).group(0) for s in pool}
        if len(pres) != 1:
            continue
        inner = {m["cid"] for m in cl}
        taken = {str((comps[c].get("designator") or {}).get("canon") or "")
                 for c in comps if c not in inner}
        if any(str(s["canon"]) in taken for s in pool):
            continue
        xs = [_cur_x(m) for m in cl]
        if any(xs[i] > xs[i + 1] + 1e-6 for i in range(len(xs) - 1)):
            stat["bails"] += 1
            continue
        labs = sorted(pool, key=lambda s: _cx(s["box"]))
        detail = []
        for m, s in zip(cl, labs):
            new = str(s["canon"])
            old = m["cur"]
            cur_slots = slots_all.get(m["cid"]) or []
            if (new == old and len(cur_slots) == 1
                    and cur_slots[0].get("box_i") == s.get("box_i")):
                continue
            comps[m["cid"]]["designator"] = dict(s)
            slots_all[m["cid"]] = [dict(s)]
            stat["changed"] += 1
            detail.append({"cid": m["cid"], "x": round(m["bx"], 1),
                           "old": old, "new": new})
        if detail:
            stat["details"].append({"n": len(cl), "fix": detail})
    return stat
