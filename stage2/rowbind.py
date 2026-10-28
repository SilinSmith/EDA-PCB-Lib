"""行内位号/值重绑（`STAGE2_ROW_BIND`，默认关）。

病（`open_GT` case1 的电阻条带，2026-10 实测，符号中心 x 从 1795 到 2065）：

    GT   1799 R20 / 1848 R19 / 1898 R16 / 1947 R17 / 1999 R18 / 2056 R15
    我们 1799 R20 / 1848 --- / 1897 R19 / 1947 R16 / 1998 R17 / 2056 R15

位号印在自己符号的**右侧**（`R19` 文字框 `[1862,845,1886,855]`，符号框右缘 ~1856），
而「逐颗捡最近文字框」在等距的排里必然往右偏一颗（`1874−1848 = 26` vs `1897−1874 = 23`），
于是一串全错位。`R18` 那块文字压根没读出来，`1848` 那颗就空着。

做法：按 `bbox_img` 中心 y 把**同类型**元件聚成行；行内所有元件都是「符号」，
其中有位号文字的才是「候选位号」。对「位号属于左边最近的符号」和「属于右边最近的符号」
两种印刷约定各算一次按 x 序一一配对的总位移（同一侧没有空位就丢弃该文字，记
`ROW_DROP_COST`），取代价小的那个落盘。只搬 `designator` / `value`，
不动框、不动引脚、不增删元件。
"""
import os
from typing import Dict, List, Optional

from envflag import env_on

ROW_BIND = env_on("STAGE2_ROW_BIND", "1")
ROW_TOL = 12.0          # 行内 y 中心容差
ROW_MIN = 3             # 行内至少这么多元件才动手
ROW_DROP_COST = 60.0    # 丢掉一个位号文字框的代价（相当于位移 60px）
ROW_MAX_DROP = 1        # 行内最多容忍丢这么多个，再多就认为不是这条约定
ROW_TYPES = frozenset(("r", "c", "l"))
ROW_GAP_MAX = 70.0      # 一颗符号到它该认领的文字框之间的允许距离
# ★ 行内「位号在符号左边还是右边」这条约定，按**整页多数票**定（默认开）。
#   旧逻辑按 `_pair` 的总 cost 选 side；一行里少读出一个位号时两边各只剩一种「最近邻」解，
#   cost 差得只有几个像素、由噪声决定 ⇒ 整排错位一格。
#   2026-11 实测（case1，6 符号 / 间距 49，位号文字压在两颗符号的接缝上）：
#     A(32px) 读到 5 个位号 ⇒ cost 选 `side=+1`（对，与 GT `R20@1796/R19@1845/R16@1893` 一致）；
#     B(64px) 读到 4 个位号 ⇒ cost 选 `side=-1`（错，整排右移一格、`R19` 丢失）。
#   两边的 cost 是 127.7 与 96.0，而**正确解**在少一个位号时 cost 必然更大 ⇒ cost 不可用。
ROW_SIDE_VOTE = env_on("STAGE2_ROW_BIND_SIDE_VOTE", "0")
ROW_VOTE_MIN = 8        # 至少这么多「有位号的件」才信投票
ROW_VOTE_FRAC = float(os.environ.get("STAGE2_ROW_BIND_VOTE_FRAC", "0.8"))  # 多数票占比阈值


def rename(doc: dict, mapping: Dict[str, str]) -> dict:
    """把交付文档里的位号键整体改名（`old → new`），连同 `pins` 与 `nets` 的顶点。

    只在 `STAGE2_ROW_BIND` 生效时由 `fusion/run.py` 在 `_assemble` **之后**调用：
    最终位号是 claim 解的产物，装配前改 `assigned` 会被 `doc.apply(…, sol, …)` 覆盖。
    """
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}
    moves = [(o, n) for o, n in mapping.items() if o in comps and o != n]
    if not moves:
        return doc
    tmp = {}
    for i, (old, new) in enumerate(moves):
        tk = "\x00rb%d" % i
        comps[tk] = comps.pop(old)
        if old in pins:
            pins[tk] = pins.pop(old)
        tmp[tk] = new
    for tk, new in tmp.items():
        comps[new] = comps.pop(tk)
        if tk in pins:
            pins[new] = pins.pop(tk)
    for v in (doc.get("nets") or {}).values():
        hg = (v.get("hyperGraph") or "").strip()
        if not (hg.startswith("(") and hg.endswith(")")):
            continue
        parts = []
        for t in [x.strip() for x in hg[1:-1].split(",") if x.strip()]:
            if "." in t:
                a, b = t.rsplit(".", 1)
                parts.append("%s.%s" % (mapping.get(a, a), b))
            else:
                parts.append(t)
        v["hyperGraph"] = "(" + ",".join(parts) + ")"
    return doc


def _cx(b) -> float:
    return (float(b[0]) + float(b[2])) / 2.0


def _cy(b) -> float:
    return (float(b[1]) + float(b[3])) / 2.0


def _rows(members: List[dict]) -> List[List[dict]]:
    out: List[List[dict]] = []
    for m in sorted(members, key=lambda d: d["by"]):
        for row in out:
            if abs(row[0]["by"] - m["by"]) <= ROW_TOL:
                row.append(m)
                break
        else:
            out.append([m])
    return [r for r in out if len(r) >= ROW_MIN]


def _pair(labels: List[dict], bodies: List[dict], side: int):
    """按 x 序一一配对；`side=+1` = 位号归它左边最近的符号。"""
    pairs, used, cost, drop = [], set(), 0.0, 0
    for lab in labels:
        cand = [b for b in bodies if b["cid"] not in used
                and ((b["bx"] < lab["lx"]) if side > 0 else (b["bx"] > lab["lx"]))]
        pick: Optional[dict] = (cand[-1] if side > 0 else cand[0]) if cand else None
        if pick is None:
            drop += 1
            pairs.append((lab["cid"], None))
            continue
        if abs(lab["lx"] - pick["bx"]) > ROW_GAP_MAX:
            drop += 1
            pairs.append((lab["cid"], None))
            continue
        used.add(pick["cid"])
        cost += abs(lab["lx"] - pick["bx"])
        pairs.append((lab["cid"], pick["cid"]))
    return pairs, used, cost + ROW_DROP_COST * drop, drop


def fix(doc: dict, a: dict) -> Dict[str, object]:
    """返回统计信息；`a["comps"][cid]` 里的 designator/value 会被就地重排。"""
    stat: Dict[str, object] = {"on": bool(ROW_BIND), "rows": 0, "moved": 0, "bails": 0}
    # ★ `STAGE2_ROW_BIND_DUMP=<path>`：把每一行的输入（位号框心 / 符号框心 / 选中的 side 与
    #   cost / 配对结果）落成 json。`print` 在管线里进不了 lane 的 stdout（`_lanes/lane*/stdout.log`
    #   里 grep `[rb]`/`[rowbind]` 都是空），要复核「换识别器后 `moved 3→1`」只能靠这个文件。
    _dump: List[dict] = []
    if not ROW_BIND:
        return stat
    comps = a.get("comps") or {}
    boxes = {}
    for c in doc.get("components") or []:
        b = c.get("bbox_img")
        if b and len(b) >= 4:
            boxes[c.get("id")] = (str(c.get("type") or ""), [float(x) for x in b[:4]])
    groups: Dict[str, List[dict]] = {}
    for cid, _rec in comps.items():
        bt = boxes.get(cid)
        if not bt or bt[0] not in ROW_TYPES:
            continue
        d = comps[cid].get("designator") or {}
        has = bool(d.get("box") and d.get("canon"))
        groups.setdefault(bt[0], []).append({
            "cid": cid, "bx": _cx(bt[1]), "by": _cy(bt[1]),
            "lx": _cx(d["box"]) if has else None,
        })
    _vote = 0
    if ROW_SIDE_VOTE:
        _pos = _neg = 0
        for cid, _rec in comps.items():
            _bt = boxes.get(cid)
            if not _bt or _bt[0] not in ROW_TYPES:
                continue
            _d = comps[cid].get("designator") or {}
            if not (_d.get("box") and _d.get("canon")):
                continue
            if _cx(_d["box"]) > _cx(_bt[1]):
                _pos += 1
            else:
                _neg += 1
        _n = _pos + _neg
        if _n >= ROW_VOTE_MIN:
            if _pos >= _n * ROW_VOTE_FRAC:
                _vote = 1
            elif _neg >= _n * ROW_VOTE_FRAC:
                _vote = -1
        stat["vote"] = _vote
        stat["vpos"] = _pos
        stat["vneg"] = _neg
    for _t, members in groups.items():
        for row in _rows(members):
            stat["rows"] += 1
            bodies = sorted(row, key=lambda m: m["bx"])
            labels = sorted([m for m in row if m["lx"] is not None], key=lambda m: m["lx"])
            _rec = {"type": _t, "n": len(row), "leb": [round(m["lx"], 1) for m in labels],
                    "beb": [round(m["bx"], 1) for m in bodies]}
            _dump.append(_rec)
            if len(labels) < ROW_MIN - 1:
                continue
            # 只在「行里确实有符号没拿到位号」时动手：那说明这一排要么漏读了一个文字框、
            # 要么错位吞掉了邻居的位号，重新按行配对才有意义。所有符号都已有位号的行
            # （绝大多数页）一律不碰 —— 这一条是 case4 掉 1.25 分换来的。
            if len(labels) >= len(bodies):
                _rec["skip"] = "all-labelled"
                continue
            best = None
            for side in ((_vote,) if _vote else (1, -1)):
                pairs, used, score, drop = _pair(labels, bodies, side)
                if best is None or score < best[0]:
                    best = (score, side, pairs, used, drop)
            if best is None or best[4] > ROW_MAX_DROP:
                stat["bails"] += 1
                continue
            score, side, pairs, used, drop = best
            _rec.update({"side": side, "cost": round(score, 1), "drop": drop,
                         "pairs": [[a_, b_] for a_, b_ in pairs]})
            if os.environ.get("STAGE2_ROW_BIND_DBG"):
                print("[rowbind] t=%s n=%d lab=%d side=%+d cost=%.1f drop=%d "
                      "labels=%s bodies=%s" % (
                          _t, len(row), len(labels), side, score, drop,
                          [round(m["lx"], 1) for m in labels],
                          [round(m["bx"], 1) for m in bodies]))
            snap = {b["cid"]: (comps[b["cid"]].get("designator"), comps[b["cid"]].get("value"))
                    for b in bodies}
            # 行内**缺号**：`R20/R19/R16/R17/R15` 缺 `R18` —— 那一块位号文字没读出来时，
            # 用它给没认领文字的符号补一个号（GT 里确实存在 `R18`，见 :104 的注释）。
            _pre, _nums = None, []
            for m in labels:
                _c = str((comps[m["cid"]].get("designator") or {}).get("canon") or "")
                _p = _c.rstrip("0123456789")
                if _p and _c[len(_p):].isdigit() and (_pre is None or _pre == _p):
                    _pre = _p
                    _nums.append(int(_c[len(_p):]))
            missing = []
            if _pre and len(_nums) >= 3:
                missing = ["%s%d" % (_pre, n)
                           for n in range(min(_nums), max(_nums) + 1) if n not in _nums]
            moved = 0
            for lab_cid, body_cid in pairs:
                if body_cid is None:
                    continue
                d, v = snap[lab_cid]
                comps[body_cid]["designator"] = d
                comps[body_cid]["value"] = v
                if lab_cid != body_cid:
                    moved += 1
            for b in bodies:                     # 行内没人认领的符号：补号，别清值
                if b["cid"] in used:
                    continue
                _d = comps[b["cid"]].get("designator")
                comps[b["cid"]]["designator"] = None
                _num = missing.pop(0) if missing else None
                if _num is not None and _d:
                    comps[b["cid"]]["designator"] = {
                        "text": _num, "canon": _num,
                        "conf": float((_d.get("conf") if isinstance(_d, dict) else 0.0) or 0.0),
                        "box": (_d.get("box") if isinstance(_d, dict) else None),
                        "synth": 1}
            stat["moved"] += moved
    _path = os.environ.get("STAGE2_ROW_BIND_DUMP")
    if _path:
        try:
            import json as _json
            with open(_path, "w", encoding="utf-8") as _f:
                _json.dump(_dump, _f, ensure_ascii=False, indent=1)
        except Exception:
            pass
    return stat
