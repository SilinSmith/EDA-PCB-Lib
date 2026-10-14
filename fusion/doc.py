"""把融合层的裁决**补丁**回文档。

⚠️ **出发点必须是 VLM 补强后的文档**，不能用 `assemble.build_doc(stage1, assigned)` 重建。
实测（9 张 GT，`work\\tools\\eval\\stage2_eval.py`）：

| 臂 | 总分 |
|---|---:|
| `stage2`（无 VLM） | 19.9804 |
| 纯重建（VLM 当候选 + 先验 + 解，`--no-llm`） | 22.3306 |
| `stage3_vlm`（VLM 直接改文档） | 22.5020 |

逐张看，**重建在 VLM 本来就强的地方掉分**：`0001 31.83→29.40`、`0005 13.14→10.43`、
`0009 31.29→28.05`、`0010 29.20→27.58`。原因不是「解错了」，而是**重建一次就丢东西** ——
`enrich` 对 `pins` 的补号/补名、对 `nets.hyperGraph` 的改名、以及 VLM **新增的元件**
（`U1` 那种整颗让 VLM 读出来的）在 `assigned` 里**根本没有**，`build_doc` 自然造不出来。

所以这一层的正确姿势是**补丁**：以 VLM 文档为底，只覆盖融合层**真正裁决过**的字段，
其余原样保留。判据很简单 —— 若某个槽位融合层没有比现状更好的证据，它就**不该被写**。

`enrich` 的改名会同步 `pins` 键与 `nets[*].hyperGraph`，本模块的 `rename_keys`
用的是**同一条正则**（`eda-pcb\\vlm\\enrich.py:487`），保证两边行为一致：

    re.sub(r"(?<![0-9A-Za-z_])%s\\." % re.escape(old), "%s." % new, hg)
"""

from __future__ import annotations

import copy
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# `python fusion/doc.py` 直接跑自检时 `sys.path[0]` 是 `fusion/` 而不是 `eda-pcb/`，
# `from fusion import ...` 会 `ModuleNotFoundError`（`claims.py` 是靠它自己那两个
# `except ImportError` 分支顺手插的路径，这里显式插一次）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from fusion import priors as PR                             # noqa: E402
from fusion.claims import FKEY_RE, vlm_keymap, _by_bbox     # noqa: E402

# 元件改名时，`Name` 里如果装的**就是旧键本身**（阻容类），要跟着改；
# 装的是型号（box 类）就不能动。
BOX_TYPES = frozenset({"box"})


def rename_keys(doc: dict, renames: Dict[str, str]) -> Dict[str, str]:
    """把 `renames`（旧键→新键）应用到 `components` / `pins` / `nets[*].hyperGraph`。

    与 `eda-pcb\\vlm\\enrich.py:481-489` 的改名逻辑逐字一致（同一条负向环视正则，
    避免 `C1` 误伤 `C15`）。返回**实际生效**的那部分（撞键的被丢掉）。
    """
    if not renames:
        return {}
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}
    # 目标键已被占用的：放弃（硬抢占会在字典里撞键、整颗元件消失）。
    # 注意 `taken` 要排除「自己也要搬走」的那些键，否则连环改名会被误拒。
    moving = set(renames)
    taken = set(comps) - moving
    keep = {}
    for old, new in renames.items():
        if old not in comps or new in taken or new in keep.values():
            continue
        keep[old] = new
    # A rejected move leaves its source key occupied. Prune moves into those
    # keys as well, otherwise the dictionary comprehension drops a component.
    while keep:
        occupied = set(comps) - set(keep)
        blocked = [old for old, new in keep.items() if new in occupied]
        if not blocked:
            break
        for old in blocked:
            del keep[old]
    if not keep:
        return {}
    for old, new in keep.items():
        cv = comps[old]
        if isinstance(cv, dict) and str(cv.get("Name") or "") == old:
            cv["Name"] = new                     # 阻容类：Name 就是键
    doc["components"] = {keep.get(k, k): v for k, v in comps.items()}
    if pins:
        doc["pins"] = {keep.get(k, k): v for k, v in pins.items()}
    # Resolve each original token once. Sequential substitutions turn A->B,
    # B->C into A->C and corrupt swaps even when the component table is right.
    pattern = re.compile(r"(?<![0-9A-Za-z_])(" +
                         "|".join(re.escape(k) for k in sorted(keep, key=len, reverse=True)) + r")\.")
    for net in (doc.get("nets") or {}).values():
        hg = net.get("hyperGraph")
        if isinstance(hg, str) and hg:
            net["hyperGraph"] = pattern.sub(lambda match: keep[match.group(1)] + ".", hg)
    return keep


def invert_keymap(base_doc: dict, stage1_doc: dict, assigned: dict) -> Dict[str, str]:
    """`cid -> 底文档里的当前键`。

    底文档可能是①`build_doc` 造的（键 = stage2 读出的位号），也可能是②VLM 补强后的
    （键可能被 `enrich` 改过名了）。所以先用 `vlm_keymap`（键 = stage2 位号）直接对，
    对不上（改名了）再用 `_by_bbox` 的框重叠兜底。
    """
    km = vlm_keymap(stage1_doc, assigned)          # 旧键 -> cid
    out: Dict[str, str] = {}
    for vkey, vc in (base_doc.get("components") or {}).items():
        vkey = str(vkey)
        if isinstance(vc, dict) and str(vc.get("type") or "").lower() not in BOX_TYPES:
            # A rebuilt non-IC document already contains the solved names.
            # An old name can now belong to a different component, so identify
            # that component by its unchanged body before consulting old keys.
            cid = _by_bbox(vc, stage1_doc) or km.get(vkey)
        else:
            cid = km.get(vkey) or (_by_bbox(vc, stage1_doc) if isinstance(vc, dict) else None)
        if cid:
            out[cid] = vkey
    return out


def _read_src(src: str) -> bool:
    """这条候选是**读出来的**吗（OCR / VLM），还是先验**编出来的**？

    ★ 这是本模块最重要的一道闸。实测（9 张 GT）：`priors` 的连号窗口 +
    `solve` 的补洞把**一颗位号都没读出来的**元件（键是装配层的占位符 `p046`）
    改名成了 `P146` —— 而 `schematic_parser.metrics.is_unkeyed` 承认的占位形式是
    `^∅\\d+$` 与 `^p\\d{3,4}$`（`metrics.py:62-72`），`P146` **两个都不是**：

    * 它配不上任何真实 GT 键（GT 键是 `C1`/`U3` 这种）；
    * 它**同时丢掉了坐标匹配**——GT 的 `∅NNN` 键正是靠「预测键未被键匹配占用」
      才走匈牙利坐标匹配的（`metrics.assign_components:327`）。

    ⇒ 把 `p046` 改成 `P146` 是**双重损失**。220 张的 200 训练集里 GT 键有
    **4034/9171 是 `∅NNN`**，这条路一破就是成片的。

    所以：**先验只能细化读数，不能发明身份。** 位号要改，必须有一个读出来的来源
    （`ocr:*` / `vlm:*`）。
    """
    s = str(src or "")
    return s.startswith("ocr") or s.startswith("vlm")


def _decided(sol: dict) -> Dict[str, str]:
    out = {}
    for sid, row in (sol.get("assign") or {}).items():
        if row:
            out[sid] = str(row[0])
    return out


def _decided_src(sol: dict) -> Dict[str, str]:
    out = {}
    for sid, row in (sol.get("assign") or {}).items():
        if row and len(row) > 2:
            out[sid] = str(row[2])
    return out


def apply(base_doc: dict, stage1_doc: dict, assigned: dict, sol: dict,
          log=None, pin_index: Optional[dict] = None) -> Tuple[dict, dict]:
    """以 `base_doc` 为底打补丁，返回 `(新文档, 报告)`。**不改任何入参。**"""
    doc = copy.deepcopy(base_doc)
    dec = _decided(sol)
    dsrc = _decided_src(sol)
    before = invert_keymap(base_doc, stage1_doc, assigned)
    rep = {"n_key": 0, "n_gain": 0, "n_value": 0, "n_partname": 0, "n_skip": 0,
           "renames": {}, "keys_same": 0, "drop": [], "n_invent": 0,
           "n_pinname": 0, "n_pinnum": 0, "n_pin_skip": 0, "n_pin_invent": 0}

    # ---- ① 位号改名（一次算完、一次应用，避免链式改名互相踩） --------------
    ren = {}
    for sid, val in dec.items():
        m = re.match(r"^comp\.([^.]+)\.key$", sid)
        if not m:
            continue
        cid = m.group(1)
        cur = before.get(cid)
        if not cur:
            rep["drop"].append([cid, val, "文档里没有这颗元件"])
            continue
        if val == cur:
            rep["keys_same"] += 1
            continue
        if not PR.DESIG_RE.match(val) or FKEY_RE.match(val):
            rep["drop"].append([cid, val, "不像位号"])
            continue
        # ★ 闸：占位键 -> 真位号，必须有**读出来的**来源
        if FKEY_RE.match(cur) and not _read_src(dsrc.get(sid)):
            rep["n_invent"] += 1
            rep["drop"].append([cid, val, "先验编的位号（来源 %s）" % dsrc.get(sid)])
            continue
        ren[cur] = val
    rep["renames"] = dict(ren)
    keep = rename_keys(doc, ren)
    rep["n_key"] = len(keep)

    # 改名会换键，`cid -> 键` 要重算
    after = invert_keymap(doc, stage1_doc, assigned)

    # ---- ② 值 / 型号 ------------------------------------------------------
    for sid, val in dec.items():
        m = re.match(r"^comp\.([^.]+)\.(value|partname)$", sid)
        if not m:
            continue
        cid, fld = m.group(1), m.group(2)
        k = after.get(cid)
        cv = (doc.get("components") or {}).get(k) if k else None
        if not isinstance(cv, dict):
            rep["n_skip"] += 1
            continue
        if fld == "value":
            old_v = cv.get("value")
            if val and str(old_v or "") != val:
                if old_v not in (None, ""):
                    # ★★ owner 规矩「可以削弱内容的置信度，但不能轻易覆盖，不然很多
                    #   关键信息都被覆盖了」—— 这里原来是**唯一判据 = 与现值不相等**、
                    #   零分数比较，被顶掉的值在交付文档里**一个字节都不留**（全仓
                    #   审计点名这是该规矩的最大违反点）。不改变结果，只留痕：
                    #   旧值 + 旧来源 + 新值 + 由谁改的，一起进 `_replaced`。
                    cv.setdefault("_replaced", []).append(
                        {"field": "value", "was": old_v,
                         "was_src": cv.get("source"), "now": val,
                         "by": str(dsrc.get(sid) or "")})
                    rep["n_value_replaced"] = rep.get("n_value_replaced", 0) + 1
                cv["value"] = val
                rep["n_value"] += 1
        else:
            # `Name` 对 box 类是型号；对阻容类是键本身，**不能**被型号覆盖
            ct = str(cv.get("type") or "").lower()
            if ct in BOX_TYPES and val and str(cv.get("Name") or "") != val:
                old_n = cv.get("Name")
                if old_n not in (None, ""):
                    cv.setdefault("_replaced", []).append(
                        {"field": "Name", "was": old_n,
                         "was_src": cv.get("source"), "now": val,
                         "by": str(dsrc.get(sid) or "")})
                    rep["n_partname_replaced"] = rep.get("n_partname_replaced", 0) + 1
                cv["Name"] = val
                rep["n_partname"] += 1

    # ---- ③ 引脚号 / 引脚名 ------------------------------------------------
    # ★ 这条路以前是**全丢**的：`apply` 只认 `comp.*` 三个槽位，而
    # `claims.from_stage2` 会产生 `pin.<cid>.<k>.number` / `.name`、`solve.write_back`
    # 也会裁决它们 —— 实测 220 张：引脚槽位 1412 个、融合层「改过」1412 个，
    # 而交付文档的 `pins` 改动 **0 个**。裁决全落在旁路产物 `_assigned/` 里，
    # 交付文档一个字节都不动。这里把它接上。
    #
    # 引脚索引 `k` **不等于**文档里的键：`build_doc` 的键是 `pin_<读出来的号>`，
    # 3 种反推法覆盖率只有 0.560 / 0.462 / 0.318（`work\tools\diag\_pinmap.py`），
    # 所以真值由 `stage2` 落到旁路文件 `<case>.pinindex.json`，这里读它。
    pin_num_ren: Dict[str, str] = {}
    for sid, val in dec.items():
        m = re.match(r"^pin\.([^.]+)\.(\d+)\.(number|name)$", sid)
        if not m:
            continue
        cid, k, fld = m.group(1), m.group(2), m.group(3)
        if not pin_index:
            rep["n_pin_skip"] += 1
            continue
        # ★ 闸：只有**读出来的**来源才能改交付文档（与位号那条闸同一判据）。
        # 引脚号的先验（`priors.py:711-713` 会给空槽派 1..N）不是证据。
        if not _read_src(dsrc.get(sid)):
            rep["n_pin_invent"] += 1
            continue
        kk = after.get(cid)
        pg = (doc.get("pins") or {}).get(kk) if kk else None
        if not isinstance(pg, dict):
            rep["n_pin_skip"] += 1
            continue
        pk = (pin_index.get(cid) or {}).get(str(k))
        pv = pg.get(pk) if pk else None
        if not isinstance(pv, dict):
            rep["n_pin_skip"] += 1
            continue
        if fld == "name":
            old_pn = pv.get("pinname")
            if val and str(old_pn or "") != val:
                if old_pn not in (None, ""):
                    pv.setdefault("_replaced", []).append(
                        {"field": "pinname", "was": old_pn,
                         "was_src": pv.get("source"), "now": val,
                         "by": str(dsrc.get(sid) or "")})
                    rep["n_pinname_replaced"] = rep.get("n_pinname_replaced", 0) + 1
                pv["pinname"] = val
                rep["n_pinname"] += 1
            continue
        # ---- 引脚号：要连**键名**一起改（官方按 `pin_<号>` 匹配）----------
        newnum = str(val or "").strip()
        oldnum = str(pk)[4:].split("_")[0]
        if not newnum.isdigit() or newnum == oldnum:
            continue
        newpk = "pin_%s" % newnum
        if newpk in pg:                      # 撞号 ⇒ 不动，别把两个引脚并成一个
            rep["n_pin_skip"] += 1
            continue
        pg[newpk] = pg.pop(pk)
        # `pinname` 本来就是「读不到名字时回退成号」（`assemble.py: nm = r["nm"] or num`），
        # 号变了但它还等于旧号，就跟着改；真读出过名字的不许覆盖。
        if str(pg[newpk].get("pinname") or "") == oldnum:
            pg[newpk]["pinname"] = newnum
        # 下一轮（或同一轮的 `.name` 槽位）要能找到新键
        pin_index.setdefault(cid, {})[str(k)] = newpk
        if kk:
            pin_num_ren["%s.%s" % (kk, oldnum)] = "%s.%s" % (kk, newnum)
        rep["n_pinnum"] += 1

    # 引脚号一改，`nets[*].hyperGraph` 里的 `<键>.<号>` 必须跟着改 ——
    # `rename_keys` 只重写**键**，不碰号，两份数据会当场对不上。
    if pin_num_ren:
        for nv in (doc.get("nets") or {}).values():
            hg = nv.get("hyperGraph") if isinstance(nv, dict) else None
            if not isinstance(hg, str) or not hg:
                continue
            toks = [t for t in hg.strip().strip("()").split(",") if t]
            out = [pin_num_ren.get(t, t) for t in toks]
            if out != toks:
                nv["hyperGraph"] = "(" + ",".join(out) + ")"
    return doc, rep


def diff(a: dict, b: dict) -> dict:
    """`a`（底文档）与 `b`（补丁后）的差异摘要，用于自检与报告。"""
    out = {"key_added": [], "key_removed": [], "n_changed": 0}
    ca, cb = a.get("components") or {}, b.get("components") or {}
    out["key_added"] = sorted(set(cb) - set(ca))
    out["key_removed"] = sorted(set(ca) - set(cb))
    for k in set(ca) & set(cb):
        for f in ("value", "Name"):
            if (ca[k] or {}).get(f) != (cb[k] or {}).get(f):
                out["n_changed"] += 1
    return out


def _demo() -> int:
    """自检：改名同步 `hyperGraph`、`Name` 跟随、阻容不被型号覆盖、撞键保护。

    这里用的是**真的** `vlm_keymap` —— 它按 `stage1` 文档里 `components[].id` 与
    `assigned` 的位号推导装配键；`_by_bbox` 则靠 `bbox_json`/`bbox_img` 的重叠兜底
    （VLM 改过名之后旧键就查不到了，只能靠框）。
    """
    doc = {
        "case": "x",
        "components": {
            "p001": {"type": "r", "Name": "p001", "bbox": [0, 0, 10, 10]},
            "p002": {"type": "box", "Name": "RP2350B", "bbox": [20, 0, 60, 40]},
            "p009": {"type": "c", "Name": "p009", "bbox": [90, 0, 100, 10]},
            "C15": {"type": "c", "Name": "C15", "value": "100nF", "bbox": [70, 0, 80, 10]},
        },
        "pins": {"p001": {"pin_1": {"pinname": "1"}},
                 "p002": {"pin_1": {"pinname": "VDD"}}},
        "nets": {"net_1": {"hyperGraph": "(p001.1,p002.3)"},
                 "net_2": {"hyperGraph": "(C15.1)", "label": "GND"}},
    }
    stage1 = {"components": [
        {"id": "c0000", "bbox_json": [0, 0, 10, 10]},
        {"id": "c0001", "bbox_json": [20, 0, 60, 40]},
        {"id": "c0002", "bbox_json": [70, 0, 80, 10]},
        {"id": "c0003", "bbox_json": [90, 0, 100, 10]},
    ]}
    assigned = {"comps": {"c0000": {"type": "r", "designator": {"text": "p001"}},
                          "c0001": {"type": "box", "designator": {"text": "p002"}},
                          "c0002": {"type": "c", "designator": {"text": "C15"}},
                          "c0003": {"type": "c"}}}
    sol = {"assign": {"comp.c0000.key": ("R7", 1.15, "vlm:enrich"),
                      "comp.c0001.key": ("U3", 1.15, "vlm:enrich"),
                      "comp.c0002.key": ("C15", 1.15, "vlm:enrich"),
                      "comp.c0003.key": ("P109", 0.5, "solve:gapfill"),
                      "comp.c0001.partname": ("W25Q128JVPIM", 1.15, "vlm:enrich"),
                      "comp.c0002.partname": ("XX", 1.15, "vlm:enrich")}}
    new, rep = apply(doc, stage1, assigned, sol)
    ok = []
    ok.append(("装配键表推对了",
               invert_keymap(doc, stage1, assigned) ==
               {"c0000": "p001", "c0001": "p002", "c0002": "C15", "c0003": "p009"}))
    ok.append(("p001 改名成 R7", "R7" in new["components"]))
    ok.append(("p002 改名成 U3", "U3" in new["components"]))
    ok.append(("★ 先验编的位号被拦下：p009 没被改成 P109",
               "p009" in new["components"] and "P109" not in new["components"]
               and rep["n_invent"] == 1))
    ok.append(("hyperGraph 同步", new["nets"]["net_1"]["hyperGraph"] == "(R7.1,U3.3)"))
    ok.append(("改名后 pins 键跟随", "R7" in new["pins"] and "U3" in new["pins"]))
    ok.append(("box 的 Name 被型号覆盖",
               new["components"]["U3"]["Name"] == "W25Q128JVPIM"))
    ok.append(("阻容的 Name 跟随新键（不被型号覆盖）",
               new["components"]["R7"]["Name"] == "R7"))
    ok.append(("同位号不重复改", "C15" in new["components"] and rep["n_key"] == 2))
    ok.append(("C15 的值未被误改", new["components"]["C15"]["value"] == "100nF"))
    ok.append(("c0002 的 partname 对阻容不生效",
               new["components"]["C15"]["Name"] == "C15"))
    ok.append(("底文档没被改", doc["components"]["p001"]["Name"] == "p001"))
    ok.append(("net_2 的 label 保留", new["nets"]["net_2"]["label"] == "GND"))
    ok.append(("diff 认得出加的键", set(diff(doc, new)["key_added"]) == {"R7", "U3"}))
    bad = [n for n, v in ok if not v]
    print("doc 自检 %d 项：%s" % (len(ok), "全部通过" if not bad else "失败 " + str(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_demo())
