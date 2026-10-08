# -*- coding: utf-8 -*-
"""证据池：把**所有读取器**的产出汇成统一的「槽位 + 候选」，并给每个槽位一个取景框。

为什么要有这一层
----------------
现在每个读取器都直接往 `assigned` 里**写一个结论**：

    stage1/detect.py    -> components[i].type
    cvparts/textbox.py  -> texts[j].bbox
    textrec/infer.py    -> texts[j].reads[role] = (text, conf)
    vlm/enrich.py       -> 直接改 assigned.comps[cid].designator.text
    stage1/pintrace.py  -> pins[cid] = [[x, y], ...]
    stage1/nets.py      -> nets[k].pins

于是**没有任何地方**能回答「这条网络该不该合并」「这个引脚号该挂在哪个桩上」
——因为结论已经被就地覆盖，候选之间的冲突丢失了。实测后果：VLM 多读出来的位号
挤掉了原本正确的框归属（220 张 `component.bbox_mismatch` **281 → 420**），
而本该拦住它的 `plausible`/`solve` 不存在。

本模块只做一件事：**把结论重新变成候选**。它不猜、不判，只负责收集与定位，
裁决交给 `solve.py`（约束）与 `llm_judge.py`（带图的 LLM）。

槽位命名（稳定契约，`priors`/`solve`/`plausible` 都按这个前缀分派）
-----------------------------------------------------------------
    comp.<cid>.key        这颗元件的位号（官方按 key 匹配元件，最值钱的一项）
    comp.<cid>.value      参数类元件的值（r/c/l/fuse/crystal_*）
    comp.<cid>.partname   非参数类元件的器件名（IC 的 Name 字段）
    comp.<cid>.type       元件的官方类型
    pin.<cid>.<k>.number  第 k 个引脚桩的**号**
    pin.<cid>.<k>.name    第 k 个引脚桩的**名**
    net.<net_key>.label   某条网络上的网络标签

图像坐标约定
------------
本模块里所有 `box` 都是 **stage1 的图像坐标（左上角原点、y 向下）**。
`assigned` 里也是图像坐标（`assemble.build_doc` 才翻 y）。
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from envflag import env_on
except ImportError:                                    # 直接跑脚本自检
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envflag import env_on

try:
    from cvparts.hypo import (CaseHypotheses, Slot, Alt,     # noqa: F401
                              MARGIN, MAX_ALTS)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cvparts.hypo import (CaseHypotheses, Slot, Alt,     # noqa: F401
                              MARGIN, MAX_ALTS)

from fusion import priors as PR                            # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                                       # eda-pcb
_REPO = _ROOT.parent                                       # 仓库根 Y:\workspace\EDA

# VLM 补强文档里的读法作为候选时的分数。CRNN 那条约 1.0（`desig_slots` 的 score
# 上限），独立的 `llm_judge` 是 1.5 —— VLM 补强排在中间：它确实比 CRNN 准
# （真值紧框上 0.95 vs 0.70），但它是**批量补读**、没有针对这一处单独设问。
VLM_ENRICH = float(os.environ.get("FUSION_VLM_SCORE", "1.15"))
# 装配层给未配上位号的元件发的占位键（`assemble.build_doc` 的 `"p%03d" % i`）：
# `p008` 这种既不是位号也不是型号，被 VLM 照抄回来时必须丢掉。
FKEY_RE = re.compile(r"^p\d{3,4}$")

# 取景框在候选框基础上再外扩多少（像素）。太小则模型看不到上下文（元件轮廓、
# 相邻引脚、导线），太大则 4 倍放大后超出端点的图像预算。
REGION_PAD = float(os.environ.get("FUSION_REGION_PAD", "24"))
# 取景框的**最小边长**：位号框常常只有十几个像素，直接切出来给 VLM 看不出是什么。
REGION_MIN = float(os.environ.get("FUSION_REGION_MIN", "120"))

# ★★ boxscan（用户 m23186 原话：「我探测到了 box 元件，那么肯定要围绕他的 box 展开
#    pin 和文字的扫描。这也是为什么需要融合层：统筹各种资源引导其他的子调度。」）
#
# 改造前的缺口：第 1 节只遍历 `assigned["desig_slots"]` —— 于是**一个位号候选都没有的
# 元件连槽位都没有**，`llm_judge`/`rescan`/`cropfix`/`audit` 四个模块全部跳过它，
# `solve.open_questions` 也问不到（零候选）。实测 220 张：629 颗没配上的 GT IC 里
# **418 颗（56.0%）**的重叠元件 `desig_slots` 是空的。
#
# 打开后：对每个 `type` **不在 `NO_DESIGNATOR` 里**、又没有取景框的元件建
# `comp.<cid>.key` 槽位 + 取景框（本体框，`set_region` 再用 `REGION_MIN=120` 兜底撑开
# 视野）。零候选槽位的「去问一遍」路径在 `solve.open_questions` 的
# `FUSION_OPEN_FREEFORM` 那里。
# ⚠ VLM 开着时这一条基本是 no-op（`_add_vlm` 已经把 `comp.<cid>.key` 覆盖到 82/83、
#   12/12、22/22）；它补的是**无 VLM**口径下整条路径消失的那一半。
#
# ============================ 实测：**判负，默认关** ============================
# 口径：`--no-vlm` + `$env:FUSION_LLM='1'`（VLM 开着时 ① 是 no-op、② 要 LLM 轮真跑，
# 所以只有这个口径量得到 boxscan 本身）。同口径配对对照（两档、官方 10GT 9 张与
# 派生 19 张）：
#   官方 9：`bs0_9` **52.48**（comp 0.7682 / netH 0.5130 / 元件 tp 167）
#           vs `bs1_9` **52.47**（0.7721 / 0.5096 / 168）      ⇒ **−0.01**
#   19 张：`bs0_19` **31.32**（0.5278 / 0.2754 / 316）
#           vs `bs1_19` **31.37**（0.5302 / 0.2762 / 318）      ⇒ **+0.05**
#   （不满足「两档都赚」门槛。）
# 机制（`_fusion_report.json` 的 `notes`，一次 4 张的现场）：
#   `{"kind":"boxscan","n_slot":0,"n_no_bbox":0,"n_skip_type":8}`（case 1）
#   `{...,"n_slot":1,...,"n_skip_type":11}`（case 4）/ `n_slot:1`（case 7、case 10）
#   ⇒ **每张图最多只建 1 个槽位**（类型闸门挡掉 6~11 个），收益/损失就是那 1 颗元件的
#   自由读数对不对：case 4 那颗读错 ⇒ **57.99 → 55.69**（comp 0.7755→0.7347，多一个
#   fp）；case 7 读对 ⇒ 57.12 → 58.37；case 10 读对 ⇒ 51.56 → 52.57。**净打平。**
# 试过的补救（也否掉了）：把自由读数的分数压到读数带以下（`FUSION_FREE_SCORE=0.5`，
#   见 `llm_judge.py`），让零候选元件只能**补空位**不能**抢位** —— 实测 `bs1f_9`
#   与 `bs1_9` **逐位相同**（52.47，case 4 仍是 55.69）⇒ **case 4 的损失不是「抢位」
#   造成的，就是那颗元件被 LLM 读错了**，压分救不了。
# ⇒ 代码保留（`FUSION_BOXSCAN=1` 可复现每一臂），**默认 0**。
BOXSCAN = env_on("FUSION_BOXSCAN", "0")

# ⚠ boxscan 的**类型闸门**：这些 type 纸上本来就没有位号（key 是伪造号 `∅N`），
# 给它们建「位号是什么？」槽位只会让 LLM **编一个** key 出来 ⇒ 元件变 fp。
# 与 `stage2/assign.py:303 NO_DESIGNATOR` 保持同一份 —— 先试着 import，
# 拿不到时退回本地副本（内容逐字相同）。
try:
    from stage2.assign import NO_DESIGNATOR as _NO_DESIGNATOR      # noqa: E402
except Exception:                                                  # noqa: BLE001
    _NO_DESIGNATOR = frozenset({"gnd", "v", "net_input", "net_short", "net_output",
                                "other", "net_bidirection", "m3\u87ba\u4e1d"})


def _f(v, d: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _box4(b) -> Optional[Tuple[float, float, float, float]]:
    if not b or len(b) < 4:
        return None
    x0, y0, x1, y1 = (_f(b[0]), _f(b[1]), _f(b[2]), _f(b[3]))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def union_box(a, b):
    a, b = _box4(a), _box4(b)
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def grow_box(b, pad: float, min_side: float = 0.0, shape=None):
    """把一个框外扩 `pad`，并保证至少 `min_side` 见方（以中心为准）。`shape=(W,H)` 时钳进图内。"""
    b = _box4(b)
    if b is None:
        return None
    x0, y0, x1, y1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    w, h = x1 - x0, y1 - y0
    if min_side > 0:
        if w < min_side:
            cx = (x0 + x1) / 2.0
            x0, x1 = cx - min_side / 2.0, cx + min_side / 2.0
        if h < min_side:
            cy = (y0 + y1) / 2.0
            y0, y1 = cy - min_side / 2.0, cy + min_side / 2.0
    if shape:
        W, H = float(shape[0]), float(shape[1])
        x0, x1 = max(0.0, min(x0, W - 1)), max(1.0, min(x1, W))
        y0, y1 = max(0.0, min(y0, H - 1)), max(1.0, min(y1, H))
    return (x0, y0, x1, y1)


@dataclass
class Region:
    """一个槽位的**取景框**（图像坐标），以及给 LLM 的取景说明。

    `box` 是最终要切给模型的窗口；`focus` 是槽位真正指着的东西（画框时高亮它，
    让模型知道该看哪儿 —— 实测不给高亮时模型会去读窗口里**别的**文字）。
    """

    box: Tuple[float, float, float, float]
    focus: Optional[Tuple[float, float, float, float]] = None
    note: str = ""
    src: str = ""

    def crop(self, bgr, scale: int = 4, focus_rect: bool = False,
             mode: str = "tight"):
        """切出给模型看的图。

        `mode="tight"`（默认）：**只切高亮框**，边距按字高给（`max(6, 0.6h)`）。
        `mode="window"`：切整个取景窗口（含上下文，但模型容易被窗口里别的字带跑）。

        ⚠ 两个实测结论（`case 0001`，见 `work/tools/diag/_judgedbg.py`）：
        1. **默认不画红框**。同一个裁片，`C13` 在无框时读对，画上 2px 红框后读成
           `C151`；`R8` 无框读对、有框读成 `R8 33`。红框反而**制造**了错误。
        2. **紧切比宽窗准**。`vlm/client.py` 的记录也是同向的（VLM 在真值紧框上
           0.95，在检出框上 0.775），窗口越大越容易被邻字带跑。
        """
        import cv2

        H, W = bgr.shape[:2]
        if mode == "window":
            base = self.box
        else:
            base = self.focus or self.box
            h = max(1.0, float(base[3]) - float(base[1]))
            base = grow_box(base, max(6.0, 0.6 * h), 0.0, shape=(W, H))
        b = grow_box(base, 0.0, 0.0, shape=(W, H))
        x0, y0, x1, y1 = (int(round(b[0])), int(round(b[1])),
                          int(round(b[2])), int(round(b[3])))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        sub = bgr[y0:y1, x0:x1].copy()
        if focus_rect and self.focus:
            f = self.focus
            fx0, fy0 = int(round(f[0])) - x0, int(round(f[1])) - y0
            fx1, fy1 = int(round(f[2])) - x0, int(round(f[3])) - y0
            cv2.rectangle(sub, (fx0, fy0), (fx1, fy1), (0, 0, 255), 2)
        if scale and scale > 1:
            sub = cv2.resize(sub, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        return sub

    def to_dict(self) -> dict:
        return {"box": [round(v, 1) for v in self.box],
                "focus": [round(v, 1) for v in (self.focus or self.box)],
                "note": self.note, "src": self.src}


class ClaimPool:
    """一张图上的全部候选证据。"""

    def __init__(self, case: str, doc: Optional[dict] = None,
                 assigned: Optional[dict] = None):
        self.case = str(case)
        self.doc = doc or {}
        self.assigned = assigned or {}
        self.hy = CaseHypotheses(self.case)
        self.regions: Dict[str, Region] = {}
        self.meta: Dict[str, dict] = {}
        self.notes: List[dict] = []
        self.bgr = None

    # ------------------------------------------------------------ 写
    def add(self, slot_id: str, value: Any, score: float = 0.0, source: str = "",
            evidence: Optional[dict] = None, question: str = "",
            context: Optional[dict] = None, region: Optional[Region] = None,
            meta: Optional[dict] = None, keep: bool = True) -> Slot:
        """加一条候选。`keep=False` 表示「只是登记元数据/取景框」，不进候选表。"""
        s = self.hy.slots.get(slot_id)
        if s is None and keep:
            s = self.hy.add(slot_id, value, score, source, evidence or {},
                            question, context)
        elif s is not None and keep:
            s = self.hy.add(slot_id, value, score, source, evidence or {},
                            question, context)
        if region is not None:
            old = self.regions.get(slot_id)
            if old is None:
                self.regions[slot_id] = region
            else:
                self.regions[slot_id] = Region(
                    box=union_box(old.box, region.box),
                    focus=union_box(old.focus, region.focus),
                    note=old.note or region.note, src=old.src or region.src)
        if meta:
            self.meta.setdefault(slot_id, {}).update(meta)
        return s if s is not None else Slot(slot_id=slot_id)

    def note(self, kind: str, **kw) -> None:
        self.notes.append(dict(kind=kind, **kw))

    def set_region(self, slot_id: str, box, focus=None, note: str = "", src: str = ""):
        shape = None
        img = self.doc.get("image") or {}
        if img.get("w") and img.get("h"):
            shape = (img["w"], img["h"])
        b = grow_box(box, REGION_PAD, REGION_MIN, shape=shape)
        if b is None:
            return
        self.regions[slot_id] = Region(box=b, focus=_box4(focus) or _box4(box),
                                       note=note, src=src)

    def grow_region(self, slot_id: str, box, note: str = "", src: str = ""):
        """只把**取景窗口**撑大，**不动高亮框**。

        高亮（`focus`）必须一直指着**文字本身**。`case 0001` 实测：把元件本体框也并进
        `focus` 之后，红框变成了整个元件符号，文字落在框外 —— 模型被要求「读红框里的字」
        却看不到字。本体只该用来扩大视野（让模型知道这是电容还是 IC），不该参与高亮。
        """
        r = self.regions.get(slot_id)
        if r is None:
            self.set_region(slot_id, box, focus=box, note=note, src=src)
            return
        shape = None
        img = self.doc.get("image") or {}
        if img.get("w") and img.get("h"):
            shape = (img["w"], img["h"])
        b = grow_box(box, 0.0, 0.0, shape=shape)
        if b is None:
            return
        self.regions[slot_id] = Region(box=union_box(r.box, b), focus=r.focus,
                                       note=r.note or note, src=r.src or src)

    # ------------------------------------------------------------ 读
    def slot_ids(self) -> List[str]:
        return sorted(self.hy.slots)

    def alts(self, slot_id: str) -> List[Alt]:
        s = self.hy.slots.get(slot_id)
        return list(s.alts) if s else []

    def best(self, slot_id: str) -> Optional[Alt]:
        s = self.hy.slots.get(slot_id)
        return s.best() if s else None

    def chosen(self, slot_id: str):
        s = self.hy.slots.get(slot_id)
        if s is None:
            return None
        return s.chosen if s.resolved() else (s.best().value if s.best() else None)

    def unresolved(self, margin: float = 0.0) -> List[Slot]:
        """没定的槽位（按最优分从低到高 —— 最不确定的排前面，重扫按这个顺序）。"""
        out = []
        for s in self.hy.slots.values():
            if s.resolved() or not s.alts:
                continue
            if len({s_.key() for s_ in s.alts}) > 1:
                out.append(s)
        return sorted(out, key=lambda s: (s.best().score if s.best() else 0.0))

    def region(self, slot_id: str) -> Optional[Region]:
        return self.regions.get(slot_id)

    def to_dict(self) -> dict:
        return {
            "case": self.case,
            "n_slots": len(self.hy.slots),
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "meta": self.meta,
            "notes": self.notes,
            "hypo": self.hy.to_dict(),
        }

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                     encoding="utf-8")

    # ------------------------------------------------------------ 回写
    def apply(self, assigned: Optional[dict] = None) -> dict:
        """把每个 `comp.<cid>.*` / `pin.<cid>.<k>.*` 槽位的选定值写回 `assigned`。

        只改**值**（`canon` / `text`），原始读法与置信度留在 evidence 里，
        所以 `decision_log` 能追溯「谁改的、凭什么改的」。
        `decided_by` 会写进 `rec["_by"]` 便于事后统计。
        """
        a = self.assigned if assigned is None else assigned
        comps = a.get("comps") or {}
        for sid in self.slot_ids():
            val = self.chosen(sid)
            if val is None:
                continue
            parts = sid.split(".")
            s = self.hy.slots.get(sid)
            by = (s.decided_by if s else None) or "?"
            if parts[0] == "comp" and len(parts) == 3:
                cid, field_name = parts[1], parts[2]
                rec = comps.get(cid)
                if rec is None:
                    continue
                if field_name == "key":
                    rec["designator"] = dict(rec.get("designator") or {},
                                             canon=val, _by=by)
                elif field_name in ("value", "partname"):
                    old = rec.get(field_name) or {}
                    rec[field_name] = dict(old, text=val, _by=by)
                elif field_name == "type":
                    rec["_type_override"] = val
                    rec["_type_by"] = by
            elif parts[0] == "pin" and len(parts) == 4:
                cid, k, field_name = parts[1], parts[2], parts[3]
                rec = comps.get(cid)
                if rec is None:
                    continue
                pmap = rec.setdefault("pins", {})
                slot = pmap.get(k)
                if slot is None:
                    slot = pmap.get(str(k)) or {}
                old = slot.get(field_name) or {}
                slot[field_name] = dict(old, text=val, _by=by)
                pmap[k] = slot
                pmap[str(k)] = slot
        return a


# --------------------------------------------------------------------------
def vlm_keymap(doc: dict, assigned: dict) -> Dict[str, str]:
    """复刻 `assemble.build_doc:515-530` 的键推导，得到 `装配键 → 内部 cid`。

    ⚠ **这一步是必须的**：`components` 在装配后的文档里是**用位号当键**的
    （`{"U3": {...}, "C19": {...}}`），而 `assigned["comps"]` 用的是内部 id
    （`c0000`）。直接拿两边对 `cid` 会**一个都对不上**，而且失败是静默的。
    """
    out: Dict[str, str] = {}
    for c in (doc.get("components") or []):
        cid = c.get("id")
        info = (assigned.get("comps") or {}).get(cid) or {}
        d = info.get("designator") or {}
        key = d.get("canon") or d.get("text")
        if key and key not in out:
            out[str(key)] = cid
    return out


def _by_bbox(vc: dict, doc: dict) -> Optional[str]:
    """键对不上时（VLM 把位号改过名 ⇒ 装配键变了），用框的重叠找同一个元件。"""
    b = _box4(vc.get("bbox"))
    if b is None:
        return None
    best, best_iou = None, 0.0
    for c in (doc.get("components") or []):
        cb = _box4(c.get("bbox_json") or c.get("bbox_img"))
        if cb is None:
            continue
        ix = max(0.0, min(b[2], cb[2]) - max(b[0], cb[0]))
        iy = max(0.0, min(b[3], cb[3]) - max(b[1], cb[1]))
        inter = ix * iy
        if inter <= 0:
            continue
        ua = ((b[2] - b[0]) * (b[3] - b[1]) + (cb[2] - cb[0]) * (cb[3] - cb[1]) - inter)
        iou = inter / ua if ua > 0 else 0.0
        if iou > best_iou:
            best, best_iou = c.get("id"), iou
    return best if best_iou >= 0.3 else None


def _add_vlm(pool: "ClaimPool", vlm: dict, comps: dict,
             keymap: Optional[Dict[str, str]] = None) -> int:
    """把 VLM 补强文档里的读法**作为候选**加进池子。

    ⚠ **位号在「键」上，不在 `Name` 里。** 这是踩过的最贵的一脚：装配后的文档
    `components` 是**用位号当键**的（`{"U3": {...}}`），`Name` 只对 **box 类（IC）**
    存器件型号、对阻容就**等于键本身**（`assemble.build_doc:534-538`）。
    所以 VLM 把 `p008` 改名成 `C3` 这件事，**只体现在键从 `p008` 变成了 `C3`**；
    照着 `Name` 读，`Name == key` 会被当成「没有新信息」丢掉 —— 实测后果：
    `case 0002` 的 component F1 从 VLM 的 **0.8333 掉回 0.4167**（那 9 张上总分
    25.00 → 15.56），因为 VLM 修好的位号一个都没进池子。

    其余三条守则（也都是踩过的坑）：

    2. **键不是 cid。** 见 `vlm_keymap` —— 先查装配键表，查不到（VLM 改了名）
       就用框重叠兜底。
    3. **兜底键必须挡住。** `stage1` 给未配上位号的元件发 `p008` 这种占位键
       （`assemble.build_doc:529`）。实测 `C3.Name = "p008"` —— VLM 把占位符照抄
       回来了，既不是位号也不是型号，丢弃。
    4. **只 `add`，不 `resolve`。** VLM 读法与 CRNN 读法在池子里**平级**，
       谁赢由 `solve` 按「唯一性 + 连号 + 形状 + 来源折扣」一起判。
    """
    n = 0
    km = keymap if keymap is not None else vlm_keymap(pool.doc, pool.assigned)
    rev = {cid: k for k, cid in km.items()}
    known = set((pool.assigned.get("comps") or {}).keys())

    def put(cid, field_name, txt, why):
        nonlocal n
        sid = "comp.%s.%s" % (cid, field_name)
        # ★★ 去重闸不能只看 `pool.best()`。`best()` 是**分数最高**那条，所以
        # 「OCR 恰好先猜到了同一个字符串」时，best 已经等于 txt，VLM 这条**更强的
        # 证据就被整条丢掉**：OCR 那条留在它原本的低分上，而这个低分随后会在
        # `solve` 的唯一性/连号竞争里输掉。实测（`_shapekeys2.py`，case 0001）：
        # 形状闸把 `US8_D+`（canon `U580`，0.825）砍掉后，`comp.c0003.key` 的
        # best 变成 OCR 的 `U1`(-0.101)，于是 VLM 那条 `U1`(`VLM_ENRICH`=1.15)
        # 被这个闸挡在门外 ⇒ 最终键从正确的 `U1` 退化成 `U11`，9 张总分
        # 22.9943 → 22.8650。判据应是「**同值的候选里是否已有 ≥ 本通道强度**的证据」。
        for a in pool.alts(sid):
            if str(a.value) == txt:
                if float(a.score) >= VLM_ENRICH:
                    return                 # 已有不弱于 VLM 的证据，不必重复登记
                break                      # 有同值但更弱的证据 -> 照加，合并时抬分
        ctype = ((pool.assigned.get("comps") or {}).get(cid) or {}).get("type")
        pool.add(sid, txt, score=VLM_ENRICH, source="vlm:enrich",
                 evidence={"来源": "vlm/enrich.py", "原字段": why},
                 question="元件 %s 的%s是什么？"
                          % (cid, "位号" if field_name == "key" else "器件型号"),
                 meta={"cid": cid, "type": ctype})
        n += 1

    for vkey, vc in (vlm.get("components") or {}).items():
        if not isinstance(vc, dict):
            continue
        vkey = str(vkey)
        cid = km.get(vkey) or _by_bbox(vc, pool.doc)
        if not cid or cid not in known:
            continue                                   # VLM 新加的元件：分配层还没有它
        ctype = vc.get("type") or ((pool.assigned.get("comps") or {})
                                   .get(cid) or {}).get("type")
        # (a) **键本身** = VLM 认定的位号
        if PR.DESIG_RE.match(vkey) and vkey != rev.get(cid):
            put(cid, "key", vkey, "键")
        # (b) `Name`：box 类上是器件型号；其余类上等于键，没有新信息
        nm = vc.get("Name")
        if nm and isinstance(nm, str):
            nm = nm.strip()
            if nm and not FKEY_RE.match(nm) and nm != vkey:
                # ★★ 器件型号与位号**同形**，不能拿 `Name` 无条件顶位号。
                # `PR.DESIG_RE` 是 `^[A-Za-z]{1,3}\d{1,4}$`，而 `MRF521`（MRF+521）、
                # `MAX232`、`BCM2835`、`STC11`、`XOSC1` 这些**器件型号**全都命中它。
                # 实测 case 0036：真实位号 `U1`（整个文档里唯一带引脚的那个元件，
                # 7 个引脚）被 `Name`=`MRF521` 顶掉 ⇒ `pin` tp 2→0、`component`
                # tp 18→17，该张 56.98 → 40.31；220 张净 −0.52（掉分 51 张 −149.6，
                # 加分 26 张 +35.3），最大的几张全是「pin tp k→0、fp k→0」。
                # 判据改成：**只有当前键还是占位符（`pNNN`）或压根没有键时**，
                # 才允许用 `Name` 当位号；其余情况 `Name` 一律走 `partname`。
                cur = str(rev.get(cid) or "")
                use_as_key = (PR.DESIG_RE.match(nm) is not None
                              and (not cur or FKEY_RE.match(cur) is not None))
                if use_as_key:
                    if nm != cur:
                        put(cid, "key", nm, "Name")
                elif len(nm) >= 4:
                    put(cid, "partname", nm, "Name")
        # (c) 参数值
        val = vc.get("value")
        if isinstance(val, str) and val.strip():
            pool.add("comp.%s.value" % cid, val.strip(), score=VLM_ENRICH,
                     source="vlm:enrich",
                     evidence={"来源": "vlm/enrich.py", "原字段": "value"},
                     question="元件 %s 的参数值是什么？" % cid,
                     meta={"cid": cid, "type": ctype})
            n += 1
        bb = vc.get("bbox")
        if bb:
            for f in ("key", "value", "partname"):
                sid = "comp.%s.%s" % (cid, f)
                if sid in pool.hy.slots and not pool.region(sid):
                    pool.set_region(sid, bb, focus=bb,
                                    note="VLM 补强的元件 %s" % cid, src="vlm:enrich")
    pool.note("vlm_ingest", n=n, n_key=len(km))
    return n


def from_stage2(case: str, doc: dict, assigned: dict, bgr=None,
                vlm: Optional[dict] = None) -> ClaimPool:
    """把一份 stage1 产物 + stage2 的 `_assigned` 变成证据池。

    这一步**不做任何裁决**：`assigned` 里已经是结论的东西，在这里降级成
    「一条来源为 `stage2:<原判据>` 的候选」；真正的比较交给 `solve`。

    `vlm` 是**已经跑过 VLM 补强的文档**（`vlm/enrich.py` 的产物 `<case>.json`）。
    它必须传进来的原因是一个实测事实：`enrich_dir` 只改 `<case>.json`，
    **`_assigned` 原样拷贝**（`enrich.py:693-700`）—— 于是 VLM 的新读法绕过了整个
    分配层，直接覆盖文档字符串。后果量得出来：`component.text_mismatch` 915→575
    （读对了），`component.bbox_mismatch` **281→420**（新读法挤掉了原本正确的框归属）。
    把它当**候选**（而不是结论）喂进池子，是让「读」和「裁决」分开的唯一办法。
    """
    pool = ClaimPool(case, doc, assigned)
    pool.bgr = bgr
    comps = {c["id"]: c for c in (doc.get("components") or [])}
    for cid, rec in (assigned.get("comps") or {}).items():
        pool.meta.setdefault("comp.%s.key" % cid, {})["type"] = \
            (rec or {}).get("type") or (comps.get(cid) or {}).get("type")

    # ---- 1 位号：desig_slots 是 stage2 已经算好的**全部候选** ----------------
    shape_dropped: Dict[str, List[str]] = {}
    for cid, cands in (assigned.get("desig_slots") or {}).items():
        c = comps.get(cid) or {}
        ctype = c.get("type") or (assigned.get("comps", {}).get(cid) or {}).get("type")
        sid = "comp.%s.key" % cid
        q = "元件 %s（类型 %s）的位号是哪一个？" % (cid, ctype)
        n_shape = 0
        for cd in cands:
            canon = cd.get("canon")
            if not canon:
                continue
            # ★★ 读数形状闸：**在原样文本上**判，不是在 `canon` 上判。
            # `canon_designator` 会把尾部形近字母换成数字（`USB_D+` -> `U580`），
            # 让一个网络名变成合法位号 —— 详见 `priors.desig_shape_ok` 的 docstring。
            # `text` 缺失时退回用 `canon` 判（老结构兼容，不能fail-closed）。
            raw = cd.get("text")
            chk = raw if str(raw or "").strip() else canon
            if PR.SHAPE_GATE and not PR.desig_shape_ok(chk):
                n_shape += 1
                if PR.SHAPE_SCORE:
                    pool.add(sid, canon, score=float(PR.SHAPE_SCORE),
                             source="ocr:designator:shape",
                             evidence={"文本": cd.get("text"), "形状闸": "压分",
                                       "置信度": round(_f(cd.get("conf")), 3)},
                             question=q, meta={"type": ctype, "cid": cid})
                continue
            pool.add(sid, canon, score=_f(cd.get("score")), source="ocr:designator",
                     evidence={"文本": cd.get("text"), "置信度": round(_f(cd.get("conf")), 3),
                               "框": [_f(v) for v in (cd.get("box") or [])],
                               "离框": round(_f(cd.get("d")), 1)},
                     question=q, meta={"type": ctype, "cid": cid})
        if n_shape:
            shape_dropped[sid] = [str(cd.get("text")) for cd in cands
                                  if not PR.desig_shape_ok(
                                      cd.get("text") or cd.get("canon") or "")]
        # ★ 取景框只设**一次**：窗口 = 全部候选框的并集，高亮 = **分数最高**那个候选的框。
        # ⚠ 这里踩过一次坑：原来在候选循环里逐个 `set_region`，而 `set_region` 是**覆盖**语义
        # ⇒ 最后一个候选（＝分数**最低**的）赢。`case 0001` 的 `c0015` 候选是
        # `C5(0.754) / C1 / C5 / C3(-1.77)`，高亮最终落在 `C3` 那个框上，而那个框里印的是
        # `R8` —— 模型老老实实答了 `R8`，是取景错了不是模型错了。
        boxed = [cd.get("box") for cd in cands if cd.get("box")]
        if boxed:
            win = [_f(v) for v in boxed[0]]
            for bx in boxed[1:]:
                win = union_box(win, [_f(v) for v in bx]) or win
            pool.set_region(sid, win, focus=[_f(v) for v in boxed[0]],
                            note="元件 %s（%s）的位号候选框" % (cid, ctype),
                            src="ocr:designator")
        # 本体框也给进去：位号经常印在本体边上，取景必须包含本体才认得出是什么。
        # ⚠ 用 `grow_region`（只撑窗口、不动高亮）—— 高亮必须留在文字上。
        pool.grow_region(sid, c.get("bbox_img"),
                         note="元件 %s（%s）的整体" % (cid, ctype),
                         src="stage1:component")

    if shape_dropped:
        pool.note("shape_gate", n=len(shape_dropped), dropped=shape_dropped)

    # ---- 1b boxscan：连位号候选都没有的元件，也要有槽位和取景框 --------------
    # 见文件顶部 `BOXSCAN` 的长注释。这里只做「建槽位 + 建框」；「去问一遍」在
    # `solve.open_questions`（零候选以前被 `continue` 挡死）。
    #
    # ⚠ 2026-10 修正：第一版**没有类型闸门**，于是 `gnd` / `v` / `net_*` / `other`
    # 这些 `stage2/assign.py:303 NO_DESIGNATOR` **故意不给位号候选**的类型也被建了
    # `comp.<cid>.key` 槽位，去问 LLM「这颗元件的位号是什么」—— 它们纸上本来就没有位号，
    # LLM 只能**编一个**出来，编出来的 key 配不上 GT ⇒ 元件变 fp。
    # 实测（官方 9 张，`--no-vlm` + `FUSION_LLM=1` 同口径配对）：
    #   无闸门 `bs1_9` **52.47**（comp 0.7721 / netH 0.5096 / 元件 tp 168）
    #   vs 对照 `bs0_9` **52.48**（comp 0.7682 / netH 0.5130 / tp 167）
    #   ⇒ **净 −0.01（打平偏负）**，且 case 4 从 57.99 掉到 55.69（comp 0.7755→0.7347）。
    if BOXSCAN:
        n_new, n_no_bb, n_skip_type = 0, 0, 0
        for cid in sorted(set(comps) | set(assigned.get("comps") or {})):
            sid = "comp.%s.key" % cid
            if pool.region(sid) is not None:
                continue                                    # 第 1 节已建过
            c = comps.get(cid) or {}
            rec = (assigned.get("comps") or {}).get(cid) or {}
            ctype = rec.get("type") or c.get("type")
            if str(ctype) in _NO_DESIGNATOR:
                n_skip_type += 1
                continue
            bb = c.get("bbox_img") or rec.get("bbox_img")
            if not _box4(bb):
                n_no_bb += 1
                continue
            if sid not in pool.hy.slots:
                pool.hy.slots[sid] = Slot(
                    slot_id=sid,
                    question="元件 %s（类型 %s）的位号是什么？" % (cid, ctype),
                    context={"type": ctype, "cid": cid, "boxscan": True})
            meta = pool.meta.setdefault(sid, {})
            meta.setdefault("type", ctype)
            meta["boxscan"] = True
            pool.set_region(sid, bb, focus=bb,
                            note="元件 %s（%s）本体：无位号候选，按 box 重扫" % (cid, ctype),
                            src="boxscan:component")
            n_new += 1
        if n_new or n_no_bb or n_skip_type:
            pool.note("boxscan", n_slot=n_new, n_no_bbox=n_no_bb, n_skip_type=n_skip_type)

    # ---- 2 参数 / 器件名 ---------------------------------------------------
    for cid, rec in (assigned.get("comps") or {}).items():
        for fld in ("value", "partname"):
            d = (rec or {}).get(fld)
            if not d or not (d.get("text") or d.get("canon")):
                continue
            sid = "comp.%s.%s" % (cid, fld)
            v = d.get("canon") or d.get("text")
            # ⚠ 这里原来是 `_f(d.get("score"), 1.0)`。但 `_assigned` 里
            # `value`/`partname` 的字段集是 `{box, conf, d, text}` —— **根本没有
            # `score`**（实测 0001：value 34 条、partname 25 条，无一例外）。
            # 于是「缺分数给满分 1.0」这条兜底把**每一条**参数/器件名读数都钉在
            # 1.0：conf 0.58 与 conf 0.96 同分，同时压过所有先验与 `solve` 的补洞。
            # 现在按置信度给分，缺 conf 才退 0.5。
            pool.add(sid, v, score=_f(d.get("conf"), 0.5),
                     source="stage2:%s" % fld,
                     evidence={"文本": d.get("text"), "置信度": round(_f(d.get("conf")), 3),
                               "框": [_f(x) for x in (d.get("box") or [])]},
                     question="元件 %s 的%s是什么？" % (
                         cid, "参数值" if fld == "value" else "器件名"),
                     meta={"cid": cid, "type": (rec or {}).get("type")})
            pool.set_region(sid, d.get("box"), focus=d.get("box"),
                            note="元件 %s 的%s" % (
                                cid, "参数值" if fld == "value" else "器件名"),
                            src="stage2:%s" % fld)

    # ---- 2b VLM 补强文档：**当候选，不当结论** ------------------------------
    if vlm:
        _add_vlm(pool, vlm, comps, keymap=vlm_keymap(doc, assigned))

    # ---- 3 引脚：号与名 ----------------------------------------------------
    pinpts = doc.get("pins") or {}
    for cid, rec in (assigned.get("comps") or {}).items():
        pmap = (rec or {}).get("pins") or {}
        n = len(pinpts.get(cid) or [])
        for k in range(max(n, len(pmap))):
            slot = pmap.get(k) or pmap.get(str(k)) or {}
            px = (pinpts.get(cid) or [[None, None]] * max(1, n))
            pt = px[k] if k < len(px) else [None, None]
            for fld in ("number", "name"):
                d = slot.get(fld)
                if not d or not d.get("text"):
                    continue
                sid = "pin.%s.%d.%s" % (cid, k, fld)
                pool.add(sid, d.get("text"), score=_f(d.get("conf"), 0.5),
                         source="ocr:pin_%s" % fld,
                         evidence={"置信度": round(_f(d.get("conf")), 3),
                                   "框": [_f(x) for x in (d.get("box") or [])]},
                         question="元件 %s 第 %d 个引脚的%s是？" % (
                             cid, k, "号" if fld == "number" else "名"),
                         meta={"cid": cid, "k": k,
                               "type": (rec or {}).get("type"),
                               "pt": [round(_f(pt[0]), 1), round(_f(pt[1]), 1)]},
                         # ⚠ 这里**必须**是 `keep=False`（自检「引脚槽位不进候选表」也
                         # 钉住了这条）。试过改成 `keep=True`（让 stage2 的引脚读数
                         # 成为证据、先验退回只补空位），220 张实测**更差**：
                         # 28.4973 → **27.6871**（pin 0.0821→0.0783、
                         # netH 0.2163→**0.1966** tp 3398→3139、fp 4477→4736）。
                         # 原因：stage2 的引脚读数本身 F1 只有 0.08 量级，让它进候选表
                         # 就压过了先验在 `solve` 里守住的连号位置，再顺着 `pin_<N>`
                         # 键与 `hyperGraph` 放大成网络错。**已回退**。
                         keep=False)          # 单候选不进候选表，等 priors 补先验
                pool.set_region(sid, [pt[0] - 12, pt[1] - 12, pt[0] + 12, pt[1] + 12]
                                if pt and pt[0] is not None else None,
                                focus=[pt[0] - 12, pt[1] - 12, pt[0] + 12, pt[1] + 12]
                                if pt and pt[0] is not None else None,
                                note="元件 %s 第 %d 脚（%s）" % (
                                    cid, k, (rec or {}).get("type")),
                                src="stage1:pin")

    # ---- 4 网络标签 --------------------------------------------------------
    for i, lb in enumerate(assigned.get("net_labels") or []):
        sid = "netlabel.%d" % i
        pool.add(sid, lb.get("text"), score=_f(lb.get("conf"), 0.5),
                 source="ocr:net_label",
                 evidence={"置信度": round(_f(lb.get("conf")), 3),
                           "框": [_f(x) for x in (lb.get("box") or [])],
                           "离框": round(_f(lb.get("d")), 1)},
                 question="这个网络标签写的是什么？", meta={"idx": i})
        pool.set_region(sid, lb.get("box"), focus=lb.get("box"),
                        note="网络标签", src="ocr:net_label")

    return pool


def load_image(doc: dict):
    """按文档里的 `image.path` 读图。

    ⚠ 文档里存的是**相对仓库根**的路径（`work\\_submit_in220\\0001.png`）。直接按
    cwd 打开会在 `cwd=eda-pcb` 时静默失败（`imread_unicode` 返回 `None`，池子没图，
    `audit`/`cropfix` 全部报 `no-image`）—— 现场踩过，所以这里依次试 cwd / eda-pcb /
    仓库根三个基准。
    """
    p = (doc.get("image") or {}).get("path")
    if not p:
        return None
    try:
        from stage2.text import imread_unicode
    except Exception:                                          # noqa: BLE001
        return None
    cands = [Path(p)]
    if not Path(p).is_absolute():
        cands += [_ROOT / p, _REPO / p]
    for c in cands:
        try:
            if c.exists():
                bgr = imread_unicode(str(c))
                if bgr is not None:
                    return bgr
        except Exception:                                      # noqa: BLE001
            continue
    return None


def from_file(case: str, stage1_path, assigned_path, bgr=None,
              vlm: Optional[dict] = None) -> ClaimPool:
    doc = json.loads(Path(stage1_path).read_text(encoding="utf-8"))
    a = json.loads(Path(assigned_path).read_text(encoding="utf-8"))
    if bgr is None:
        bgr = load_image(doc)
    return from_stage2(case, doc, a, bgr=bgr, vlm=vlm)


# ------------------------------------------------------------------ 自检
def _demo() -> int:
    doc = {
        "image": {"w": 800, "h": 600, "path": ""},
        "components": [
            {"id": "c0", "type": "r", "bbox_img": [100, 100, 140, 110]},
            {"id": "c1", "type": "box", "bbox_img": [300, 100, 400, 300]},
        ],
        "pins": {"c0": [[90, 105], [150, 105]], "c1": [[290, 150], [290, 250]]},
    }
    assigned = {
        "comps": {
            "c0": {"type": "r", "designator": {"text": "R7", "canon": "R7", "conf": .9},
                   "value": {"text": "10K", "conf": .9},
                   "pins": {0: {"number": {"text": "1", "conf": .9}, "name": None},
                            1: {"number": {"text": "2", "conf": .9}, "name": None}}},
            "c1": {"type": "box", "designator": {"text": "U1", "canon": "U1", "conf": .8},
                   "partname": {"text": "MSP432E4", "conf": .7},
                   "pins": {0: {"number": {"text": "12", "conf": .8},
                                "name": {"text": "VDD", "conf": .8}}}},
        },
        "desig_slots": {
            "c0": [{"canon": "R7", "text": "R7", "conf": .9, "score": 1.2,
                    "box": [100, 82, 118, 94], "d": 6.0},
                   {"canon": "R1", "text": "Rl", "conf": .5, "score": .6,
                    "box": [104, 82, 122, 94], "d": 6.0}],
            "c1": [{"canon": "U1", "text": "U1", "conf": .8, "score": 1.0,
                    "box": [330, 120, 352, 134], "d": 0.0}],
        },
        "net_labels": [{"text": "VBUS", "conf": .9, "box": [152, 8, 170, 20], "d": 3.0}],
    }
    p = from_stage2("t", doc, assigned)
    ok = []
    ok.append(("位号槽位", "comp.c0.key" in p.hy.slots and len(p.alts("comp.c0.key")) == 2))
    ok.append(("引脚槽位不进候选表", "pin.c0.0.number" not in p.hy.slots))
    ok.append(("引脚 meta", p.meta.get("pin.c0.0.number", {}).get("pt") == [90.0, 105.0]))
    ok.append(("取景框已建", p.region("comp.c0.key") is not None
               and p.region("comp.c0.key").box[0] < 100.0))
    ok.append(("网络标签槽位", p.chosen("netlabel.0") == "VBUS"))
    # 回写
    p.hy.resolve("comp.c1.key", "U2", by="test")
    a2 = p.apply()
    ok.append(("回写位号", a2["comps"]["c1"]["designator"]["canon"] == "U2"))
    ok.append(("回写保留原读法", a2["comps"]["c1"]["designator"]["text"] == "U1"))
    bad = [n for n, c in ok if not c]
    print("claims 自检 %d 项：%s" % (len(ok), "全部通过" if not bad else "失败"))
    for n in bad:
        print("  FAIL", n)
    print("  槽位数 %d，取景框 %d，笔记 %d" % (len(p.hy.slots), len(p.regions), len(p.notes)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_demo())
