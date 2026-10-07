# -*- coding: utf-8 -*-
"""合理性检查 —— workflow.html 右侧的 `plausible` 节点。

**它和 `conflict` 的分工**（用户 m12477 的原话是这一层的由来）：

- `conflict` 问的是「**两个来源说法不一致**」；
- `plausible` 问的是「**这个结果整体讲不讲得通**」。

差别在于后者能发现**所有来源都一致地错**的情况。最典型的例子就是用户自己举的
**VCC 与 GND 短接**：如果一条网络同时挂着电源轨和地，每个读数单独看都没有任何矛盾，
`conflict` 一句话都说不出来；但整张图的网络图已经错了 —— 这种错必须靠「物理上讲不讲得通」
来发现，而不是靠「谁和谁不一致」。

用户 m12477 对它的三条要求，逐条落到这里的三种检查：

1. **结构上**：『假如 LLM 发现现有的体系中 VCC 和 GND 短接（only LLM 知道这意味着什么）
   那很明显这个电路识别错了，要在里面找错误点』 ⇒ `net_short`：电源轨与地同网。
2. **内容上**：『或者一排电容，他们都是 10nF，但是有一个是 5uF，LLM 需要判断到底是因为
   电路图本身的性质所以电容不一样，还是识别错了』 ⇒ `value_outlier`：**弱信号**，
   只报不判 —— 电源滤波电容本来就是 10uF 级，离群不代表错。
3. **时间-精度-token 平衡**：『如果它不能带来足够的时间-精度-token 平衡收益，那就算了』
   ⇒ 每条可疑项都带 `severity` 与 `hit` 预期，`rescan` 按优先级决定值不值得重扫。

本模块**只报疑点、不改结论**。改与不改由 `llm_judge`/`rescan` 决定。
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                                  # eda-pcb
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fusion import claims as CL                        # noqa: E402
from fusion import priors as PR                        # noqa: E402

# 一条网络里同时出现「地」和「电源」=> 整段网络图错了，值 0.9
SEV_SHORT = float(os.environ.get("FUSION_SEV_SHORT", "0.90"))
# 一条网络里同时出现两条**不同**的电源轨（VCC 与 3V3 是同一类，VCC 与 VBAT 不是）
SEV_RAIL = float(os.environ.get("FUSION_SEV_RAIL", "0.60"))
# 位号重复/类型不符
SEV_KEY = float(os.environ.get("FUSION_SEV_KEY", "0.70"))
# 参数离群（弱信号，必须交判）
SEV_VAL = float(os.environ.get("FUSION_SEV_VAL", "0.25"))
# 同一类型元件里，一个值要离群到多少倍才报
VALUE_RATIO = float(os.environ.get("FUSION_VALUE_RATIO", "4.0"))
# 同 (类型, 单位) 至少要几颗才做离群比较（少于这个数，中位数本身就没意义）
VALUE_MIN_N = int(os.environ.get("FUSION_VALUE_MIN_N", "4"))
# 少于这么多引脚的网算「孤网」（真图里确实有，所以只是弱信号）
MIN_NET_PINS = int(os.environ.get("FUSION_MIN_NET_PINS", "2"))

VAL_NUM_RE = re.compile(r"^([0-9.]+)\s*([a-zA-Zµμ]*)$")


# ------------------------------------------------------------------ 工具
def _f(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def parse_value(text: str):
    """把 `4.7uF` / `10k` / `100nF` 拆成 `(数字, 单位)`；拆不开返回 `None`。

    ⚠ **绝不做单位换算**：官方 Q&A Q9 明确「电容 `uF`/`μF` 以图片为准」，
    `100nF` 不许写成 `0.1uF`。这里只用来比大小，不用来改写。
    """
    m = VAL_NUM_RE.match((text or "").strip().replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(1)), m.group(2).lower()
    except ValueError:
        return None


def _scale(unit: str) -> float:
    """同族单位之间的量级（只用于比较，不写回）。"""
    return {"": 1.0, "f": 1.0, "uf": 1e-6, "µf": 1e-6, "μf": 1e-6, "nf": 1e-9,
            "pf": 1e-12, "h": 1.0, "uh": 1e-6, "nh": 1e-9,
            "r": 1.0, "k": 1e3, "m": 1e6, "ohm": 1.0, "v": 1.0,
            "mv": 1e-3, "a": 1.0, "ma": 1e-3}.get(unit, 1.0)


def _si(text: str) -> Optional[float]:
    p = parse_value(text)
    if p is None:
        return None
    n, u = p
    return n * _scale(u)


def net_labels_of(pred: dict) -> Dict[str, List[str]]:
    """从装配好的文档里取「网络 → 它挂着的标签」。"""
    out: Dict[str, List[str]] = {}
    for key, n in (pred.get("nets") or {}).items():
        labs = []
        for f in ("label", "labels", "net_label", "name"):
            v = n.get(f)
            if isinstance(v, str) and v.strip():
                labs.append(v.strip())
            elif isinstance(v, (list, tuple)):
                labs.extend(str(x).strip() for x in v if str(x).strip())
        out[key] = labs
    return out


# ------------------------------------------------------------------ 检查
def check(pool: CL.ClaimPool, pred: Optional[dict] = None) -> List[dict]:
    """返回可疑项清单（按 `severity` 降序）。

    每项：`{kind, severity, why, slots, nets, hint, need_llm}`。
    `slots` 是**可以去问/去重扫**的槽位；`hint` 告诉 `rescan` 该往哪儿看。
    """
    sus: List[dict] = []
    comps = (pool.assigned or {}).get("comps") or {}
    meta_all = pool.meta or {}

    # ---- ① 位号：同一张图里唯一 / 前缀与类型相符 ----------------------------
    by_key: Dict[str, List[str]] = defaultdict(list)
    for sid in pool.slot_ids():
        if not sid.startswith("comp.") or not sid.endswith(".key"):
            continue
        v = pool.chosen(sid)
        if v:
            by_key[str(v)].append(sid)
    for v, sids in sorted(by_key.items()):
        if len(sids) > 1:
            sus.append({
                "kind": "key_duplicate", "severity": SEV_KEY, "need_llm": True,
                "why": "位号 %s 被 %d 个槽位同时占用：%s"
                       % (v, len(sids), ", ".join(sids)),
                "slots": sids, "nets": [],
                "hint": "至少有一个读错了。把每个槽位各自的读数切图重问一次，取两两不同的解。",
            })
    for sid in pool.slot_ids():
        if not sid.startswith("comp.") or not sid.endswith(".key"):
            continue
        v = pool.chosen(sid)
        if not v:
            continue
        ctype = (meta_all.get(sid) or {}).get("type") or ""
        if ctype and not PR.key_fits_type(str(v), ctype):
            sus.append({
                "kind": "key_type_mismatch", "severity": SEV_KEY * 0.8, "need_llm": True,
                "why": "元件类型 %s（%s）配上了 %s 这样的位号"
                       % ((meta_all.get(sid) or {}).get("cid"), ctype, v),
                "slots": [sid], "nets": [],
                "hint": "要么类型判错、要么位号读错。位号是印在纸上的确定字符串，优先怀疑类型。",
            })

    # ---- ② 参数离群（弱信号，只报不判） ------------------------------------
    # ⚠ 必须**按 (类型, 单位) 分组**：`16V` 和 `100nF` 都挂在电容上，混在一起比大小
    # 会得到一堆纯噪声（第一版按 ctype 分组，`case 0001` 一口气报了 10 条全是假的）。
    groups: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = defaultdict(list)
    for sid in pool.slot_ids():
        if not sid.startswith("comp.") or not sid.endswith(".value"):
            continue
        v = pool.chosen(sid)
        if not v:
            continue
        parsed = parse_value(str(v))
        si = _si(str(v))
        if parsed is None or si is None or si <= 0:
            continue
        cid = sid.split(".")[1]
        ctype = (meta_all.get(sid) or {}).get("type") or \
            ((comps.get(cid) or {}).get("type") or "?")
        groups[(ctype, parsed[1])].append((sid, str(v), si))
    for (ctype, unit), rows in sorted(groups.items()):
        if len(rows) < VALUE_MIN_N:
            continue
        vals = sorted(r[2] for r in rows)
        med = vals[len(vals) // 2]
        if med <= 0:
            continue
        for sid, txt, si in rows:
            r = si / med if si > med else med / si
            if r >= VALUE_RATIO:
                sus.append({
                    "kind": "value_outlier", "severity": SEV_VAL, "need_llm": True,
                    "why": "同为 %s 的 %d 颗%s里，%s 的值 %s 是中位数 %g 的 %.1f 倍"
                           % (ctype, len(rows), unit or "（无单位）", sid, txt, med, r),
                    "slots": [sid], "nets": [],
                    "hint": "电源滤波电容本来就比旁路电容大两个量级 ⇒ **可能是真设计**。"
                            "只把它交给人/模型判，绝不自动改（官方是字符串严格匹配）。",
                })

    # ---- ③ 网络：电源与地同网 / 两条不同电源轨同网 --------------------------
    #    这是用户举的那个例子，也是全库最强的结构信号（gnd 2229 + v 1117 个标签）。
    if pred is not None:
        for key, labs in net_labels_of(pred).items():
            cls = [PR.net_class(x) for x in labs]
            if not cls:
                continue
            has_gnd = "ground" in cls
            pows = [x for x, c in zip(labs, cls) if c == "power"]
            if has_gnd and pows:
                sus.append({
                    "kind": "net_short", "severity": SEV_SHORT, "need_llm": False,
                    "why": "网络 %s 上同时挂着地 %s 和电源 %s —— 物理上不可能"
                           % (key, [x for x, c in zip(labs, cls) if c == "ground"], pows),
                    "slots": [], "nets": [key],
                    "hint": "一条短接意味着**整段**网络图错了，不是补一个元件能修的。"
                            "该段上的标签、引脚归属、连线合并全部重新审。",
                })
            elif len(set(pows)) > 1:
                sus.append({
                    "kind": "net_rail_merge", "severity": SEV_RAIL, "need_llm": False,
                    "why": "网络 %s 上同时挂着两条不同的电源轨 %s" % (key, sorted(set(pows))),
                    "slots": [], "nets": [key],
                    "hint": "两条轨被错误地并成一条，或其中一个是读错的信号名。",
                })

        # 孤网：真图里确实有单引脚网，所以只是弱信号
        for key, n in (pred.get("nets") or {}).items():
            pins = n.get("pins") or n.get("nodes") or []
            if len(pins) < MIN_NET_PINS:
                sus.append({
                    "kind": "net_isolated",
                    "severity": SEV_VAL, "need_llm": False,
                    "why": "网络 %s 只有 %d 个引脚" % (key, len(pins)),
                    "slots": [], "nets": [key],
                    "hint": "可能是引脚没检出来（`pin.missing` 是当前最大的一项缺口），"
                            "也可能是连线追踪断了。",
                })

    # ---- ④ 引脚：元件有引脚却一个号都没有 -----------------------------------
    for cid, rec in comps.items():
        pins = rec.get("pins") or {}
        if not pins:
            continue
        n_num = sum(1 for p in pins.values() if (p or {}).get("number"))
        if n_num == 0:
            sus.append({
                "kind": "pins_unread", "severity": SEV_VAL, "need_llm": True,
                "why": "元件 %s 有 %d 个引脚，一个号都没读出来" % (cid, len(pins)),
                "slots": ["pin.%s.%s.number" % (cid, k) for k in pins],
                "nets": [],
                "hint": "引脚号是纯数字，CRNN 只需要看出 1..N 的连号；"
                        "连号先验（`prior:pinrun`）通常已经补上，补不上就是引脚点本身错了。",
            })

    sus.sort(key=lambda d: -_f(d.get("severity")))
    return sus


# ------------------------------------------------------------------ 自检
def _demo() -> int:
    """合成一份「VCC 与 GND 短接」的最小文档，确认三类检查都报得出来。"""
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
        "c0001": {"designator": {"text": "C1", "canon": "C1"},       # 故意重号
                  "value": {"text": "10nF"}, "partname": None, "pins": {}},
        "c0002": {"designator": {"text": "C3", "canon": "C3"},
                  "value": {"text": "100nF"}, "partname": None,      # 故意离群（同单位 nf）
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

    pred = {"nets": {"0": {"pins": [["c0000", 0], ["c0001", 0]], "label": "GND"},
                     "1": {"pins": [["c0002", 0]], "label": "VCC"}}}
    # 让 GND 网同时挂上 VCC
    pred["nets"]["0"]["labels"] = ["GND", "VCC"]
    sus = check(pool, pred)
    kinds = {s["kind"] for s in sus}
    chk("报出位号重复", "key_duplicate" in kinds)
    chk("报出参数离群", "value_outlier" in kinds)
    chk("报出电源短接", "net_short" in kinds)
    chk("报出孤网", "net_isolated" in kinds)
    chk("报出引脚无号", "pins_unread" in kinds)
    chk("按 severity 降序", all(_f(sus[i]["severity"]) >= _f(sus[i + 1]["severity"])
                                for i in range(len(sus) - 1)))
    chk("短接排在最前", sus and sus[0]["kind"] == "net_short")
    chk("net_short 带 nets", sus and sus[0]["nets"] == ["0"])
    chk("value_outlier 只报不判", any(s["kind"] == "value_outlier" and s["need_llm"]
                                       for s in sus))
    chk("key_duplicate 指向两个槽位",
        any(s["kind"] == "key_duplicate" and len(s["slots"]) == 2 for s in sus))
    chk("parse_value 不换算单位", parse_value("4.7uF")[0] == 4.7
         and parse_value("100nF")[1] == "nf")
    chk("_si 只用于比较", abs(_si("1uF") / _si("1nF") - 1000.0) < 1e-6)

    print("plausible 自检 %d 项：%s" % (total, "全部通过" if ok == total else
                                        "%d 项未过" % (total - ok)))
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(_demo())
