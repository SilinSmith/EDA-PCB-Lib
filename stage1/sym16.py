# -*- coding: utf-8 -*-
"""sym16 小符号检测器的两条**默认关**臂：S1 新增提案（add-only）／S2 已有件精修（refine）。

    · `STAGE1_SYM16=1`     —— S1：把 sym16 在页面上检出、且与既有件都不重复的符号框
      作为**无名新件**追加进 `comps`（落点与 `stage1/bigic.py:add_components` 相同）。
    · `STAGE1_SYM16_FIX=1` —— S2：对**已存在**的件，用 sym16 在同位置的同型框替换
      `bbox_img`/`bbox_json`（更贴墨迹）；另允许在 sym16 置信度更高时改写 `type`。
      两条都逐件留账（`doc["stats"]["sym16_fix"]`）。

为什么只有这两条路能吃到分（判分机理，`schematic_parser/metrics.py`）
--------------------------------------------------------------------
`assign_components` 第 1 步只按**位号同名**配（`:335` `for key in set(gc) & set(pc)`），
所以新增的「无名件」永远配不上 GT 的 `U1/R1/LED1` 这类有名件；第 2 步坐标匹配的 GT
侧只取 **unkeyed**（`:351` `left_g = [k for k in gc if k not in gt_used and is_unkeyed(k)]`），
pred 侧可以是任意未用件，但 `:380` `if cost_ij >= cfg.coord_type_penalty: continue`
要求**类型相等**。⇒ 新增无名件只有在「type 恰好等于某个 GT `∅NNN` 件的 type」时才可能
变成 tp，否则必然是 `component.extra`（+1 fp）。这就是 `EMIT_TYPES` 的来源。

权重：`yolo_part/symds/data.yaml` 的 16 类、imgsz 1280（`assets.sym16_weights()`）。
产品检测器 `assets.yolo_weights()` 与它的调用参数一律不碰。
"""
from __future__ import annotations

import inspect
import os
import sys

try:                                    # 作为包导入（python -m stage1 / submission\run.py）
    from envflag import env_on      # noqa: F401  （门控在 pipeline.py 读，这里只留 import 面）
except ImportError:                     # 直接跑脚本：
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envflag import env_on      # noqa: F401

SRC = "yolo:sym16"

# sym16 的训练 imgsz（`yolo_part/symds` 的 1280）。产品线是 1600 —— 两套权重各用各的，
# 这里不继承 `submission/run.py:115` 的 `-imgsz`（那是给 pcb43 权重的）。
IMGSZ = int(os.environ.get("STAGE1_SYM16_IMGSZ", "1280") or 0)

# 16 类 -> 管线既有 `type` 字符串（左列取自 `yolo_part/symds/data.yaml` 的 classes 顺序）。
# 依据：① 与 GT `type` 字面相同的直接同名；② `block` 保留原名（GT 里 `block` 独立存在，
# 池 30 案 3 例、全 unkeyed；`metrics.canon_type` 不做 `block->box` 归一化）；
# ③ `other` 同名（池 30 案 12 例、全 unkeyed）；④ `pin` / `testpoint` 同名（GT 里都有）。
SYM2TYPE = {
    "led": "led",
    "switch": "switch",
    "gnd": "gnd",
    "testpoint": "testpoint",
    "pin": "pin",
    "v": "v",
    "d": "d",
    "net_input": "net_input",
    "net_output": "net_output",
    "net_bidirection": "net_bidirection",
    "block": "block",
    "l": "l",
    "r": "r",
    "c": "c",
    "box": "box",
    "other": "other",
}

# 只允许**生成**这些 type 的新件（S1）。依据 = 判分机理 + 13 页实测对表：
#   · `gnd` / `v`：13 页 GT 里 gnd 83/83、v 51/51 全是 `∅NNN`；池 30 案 gnd、v 同样全 unkeyed。
#     实测也只有这两类有命中：sym16 在 13 页检出 gnd 90 / v 19，去重后落进 GT unkeyed 件
#     四角≤20px 的分别是 17 / 2。
#   · 端口符号三兄弟（`stage1/symbox.py:61` `PORT_TYPES`）在池 30 案的 GT 里同样全 unkeyed
#     （net_input 21/21、net_output 4/4、net_bidirection 6/6），**但 13 页上 sym16 对端口的
#     命中是 0**（检出 5 个 net_input，去重后落进 GT 的 0 个；实际采纳后 5/5 全是
#     `component.extra`）⇒ 默认不收，宁缺勿滥。
# 其余 type（c/r/box/d/led/l/switch）在 GT 里**没有一个是 unkeyed**（13 页 0 个、池 0 个）
# ⇒ 生成它们只可能是 fp（实测 sym16 在这几类上会吐出 43+51+… 个与既有件不重复的框）。
EMIT_TYPES = frozenset(("gnd", "v"))

# 类型改写（S2）允许改到的目标类型：即「管线本来就认识、且 S1 也允许生成」的那些，
# 加上 symbox 的端口族。其余类型（pin/testpoint/block/other）val 实例数 1~12、mAP≈0，
# 一律不许改过去（`STAGE1_SYM16_TYPE` 只管开/关，不改这个白名单）。
TYPE_TARGETS = EMIT_TYPES

_MODEL = None
_MCFG = None


def dedup_iou() -> float:
    """去重阈：**直接引用** `stage1/bigic.py:212` `add_components(..., iou_skip=0.35)`
    的默认实参，不另立常数（两臂共用同一个数，改一处两边同步）。"""
    from stage1 import bigic as _bigic
    return float(inspect.signature(_bigic.add_components).parameters["iou_skip"].default)


def dedup_center() -> float:
    """sym16 追加件的中心距闸（px）。默认 20.0，可用 `STAGE1_SYM16_MIN_CD` 覆盖。

    **为什么是 20 而不是 `DEDUP_PX`（12.0）**：sym16 的 gnd 框是「整颗」，yolo 的是内框，
    同一颗符号两框中心可差 12~19px。离线扫（13 页，欧氏中心距闸，插入 `_Pred.json` 后重打分，
    基線 86.3974）：
      cd 12 → 85.8868（−0.51，插 12 个，page5 −3.70 / page11 −1.81 / page12 −1.47）
      cd 16 → 86.4341（+0.04，插 4 个，page12 −0.51）
      cd 20 → **86.4735（+0.0761，只插 3 个，全是 case1 的真漏检 ∅323/∅458/∅193，别页 0 损伤）**
      cd 24 → 同 cd 20
    ⇒ 12px 会把「已被既有件配上对的同一颗符号」再追加一次（只换 `component.extra`），20px 才干净。
    """
    try:
        v = float(os.environ.get("STAGE1_SYM16_MIN_CD", "20") or 20)
    except ValueError:
        v = 20.0
    return v if v > 0 else 0.0


# ------------------------------------------------------------------ 推理
def _detector(conf, imgsz=None):
    """同一个进程里只加载一次；权重/imgsz/conf/device 变了就重建（同 `pipeline.py:585-596`）。"""
    global _MODEL, _MCFG
    from assets import sym16_weights
    w = sym16_weights()
    cfg = (str(w), int(imgsz or IMGSZ), float(conf), "cpu")
    if _MODEL is not None and _MCFG == cfg:
        return _MODEL
    from ultralytics import YOLO
    _MODEL = YOLO(w)
    _MCFG = cfg
    return _MODEL


def raw_dets(bgr, conf=None, imgsz=None):
    """跑一遍 sym16，返回 `[{"cls", "score", "box_img"}]`（图像坐标，左上原点）。

    故意走 `ultralytics` 的**原始输出**，不经 `stage1/detect.py` 的 `Detector.detect`：
    后者会给每个框贴「图章」模板（`stamp_mod.snap_box(box, name)`，按**原类名**取模板），
    而 sym16 的 `net_input`/`net_bidirection`/`other` 这些类名在产品图章表里没有条目，
    贴章会抛/退化成未知模板。两臂要的就是检测器自己回归出来的整颗符号框。
    """
    return _predict(bgr, conf, imgsz)


def _predict(bgr, conf=None, imgsz=None):
    m = _detector(conf, imgsz)
    names = m.names
    names = [str(n) for n in names.values()] if isinstance(names, dict) else [str(n) for n in names]
    r = m.predict(bgr, imgsz=int(imgsz or IMGSZ), conf=conf, device="cpu", verbose=False)[0]
    out = []
    if r.boxes is None or not len(r.boxes):
        return out
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                       r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [float(t) for t in b]
        ci = int(c)
        out.append({"cls": names[ci] if ci < len(names) else str(ci),
                    "score": float(s), "box_img": [x1, y1, x2, y2]})
    out.sort(key=lambda d: -d["score"])
    return out


# ------------------------------------------------------------------ S1 新增提案
def add_components(bgr, comps, hyps, conf=None, imgsz=None, dets=None) -> dict:
    """把 sym16 检出的**无名新件**追加进 `comps`（add-only，不动任何既有件）。

    与 `stage1/bigic.py:add_components` 同构：同一位置、同一个去重阈、同一个 `hyps.add`
    登记、同一种 key=None 的件（`assemble.build_doc` 会给它发 `pNNN` 兜底键）。
    """
    from stage1.bigic import _iou              # 复用既有 IoU 实现，不写第二份
    from stage1.detect import DEDUP_PX          # 同型同源判同的中心距（detect.py:54）
    from stage1.detect import to_json_box

    H = int(bgr.shape[0])
    thr = dedup_iou()
    dets = raw_dets(bgr, conf, imgsz) if dets is None else dets
    have = [list(c["bbox_img"]) for c in comps if c.get("bbox_img")]
    used_ids = {str(c.get("id")) for c in comps}
    skip = {}
    added, considered = [], []
    for d in dets:
        t = SYM2TYPE.get(d["cls"])
        if t is None:                         # 数据里有、映射表里没有（换权重时的护栏）
            skip["unmapped"] = skip.get("unmapped", 0) + 1
            continue
        if t not in EMIT_TYPES:               # 只计数，不生成（见模块顶注）
            skip[t] = skip.get(t, 0) + 1
            continue
        box = [round(float(v), 2) for v in d["box_img"]]
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            skip["degenerate"] = skip.get("degenerate", 0) + 1
            continue
        # 最近既有件（IoU 最大 / 中心距最小）—— 只用于审计，不参与判据
        _bv, _bc, _bid = 0.0, 1e18, None
        for c in comps:
            cb = c.get("bbox_img")
            if not cb:
                continue
            v = _iou(box, cb)
            cd = (((box[0] + box[2]) - (cb[0] + cb[2])) ** 2
                  + ((box[1] + box[3]) - (cb[1] + cb[3])) ** 2) ** 0.5 / 2.0
            if v > _bv:
                _bv, _bid = v, c.get("id")
            if cd < _bc:
                _bc = cd
        rec = {"cls": d["cls"], "type": t, "score": round(float(d["score"]), 4),
               "box": box, "iou_max": round(_bv, 3), "cd_min": round(_bc, 1), "near": _bid}
        if any(_iou(box, hb) >= thr for hb in have):
            skip["dup"] = skip.get("dup", 0) + 1
            rec["take"] = "dup"
            considered.append(rec)
            continue
        # 第二道闸：中心距 < `DEDUP_PX` 也判同。**为什么光有 IoU 不够**：本步跑在
        # `symbox.grow` 之前，`comps` 里的符号框还是 yolo 的小框，同一颗 gnd 上用
        # sym16 的整颗大框去算 IoU 只有 0.2~0.34（实测 41 个采纳件里 35 个如此），
        # 于是 IoU 门几乎不设防 ⇒ 追加出来的全是「已被既有件配上对的同一颗符号」，
        # 只换来 `component.extra`。`DEDUP_PX`（`stage1/detect.py:54`，12.0）本来就是
        # 既有跨来源「同一个东西」的中心距判据，直接引用，不另立常数。
        if _bc < dedup_center():
            skip["near"] = skip.get("near", 0) + 1
            rec["take"] = "near"
            considered.append(rec)
            continue
        i = 0
        while "s%04d" % i in used_ids:
            i += 1
        cid = "s%04d" % i
        used_ids.add(cid)
        score = round(float(d["score"]), 4)
        hyps.add(f"{cid}.type", t, score, source=SRC,
                 evidence={"cls": d["cls"], "iou_skip": thr},
                 question=f"{cid}（sym16 检出的 {d['cls']}）是元件吗？",
                 context={"cls": d["cls"], "score": score})
        comps.append({
            "id": cid, "type": t, "proposal": t, "score": score,
            "source": SRC, "subtype": "sym16",
            "bbox_img": box,
            "bbox_json": [round(float(v), 2) for v in to_json_box(box, H)],
            "bbox_raw_img": list(box),
            "center_img": [round((box[0] + box[2]) / 2, 2), round((box[1] + box[3]) / 2, 2)],
            "key": None, "n_alts": 1, "pending": False,
        })
        have.append(box)
        added.append(cid)
        rec["take"] = cid
        considered.append(rec)
    return {"on": True, "imgsz": int(imgsz or IMGSZ), "n_dets": len(dets), "dedup_px": DEDUP_PX,
            "n": len(added), "ids": added, "skipped": skip, "considered": considered}


# ------------------------------------------------------------------ S2 已有件精修
def refine_components(bgr, comps, conf=None, imgsz=None, type_change=True, dets=None) -> dict:
    """用 sym16 的同型框替换既有件的 `bbox_img`/`bbox_json`（并可选改写 `type`）。

    调用点在 `symbox.grow` **之后**、`doc` 组装之前 —— 只改**交付框**：引脚位置
    （`comp_pins`）与导线骨架都已在更早的步骤定下来，本函数不碰它们，把影响面锁在
    元件通道。

    合并规则（`STAGE1_SYM16_FIX_MODE`）：
      · `sym`（默认）—— 直接换成 sym16 回归出来的框。sym16 是在「整颗符号」的 GT 框上
        训的（`yolo_part/symds`），13 页离线复算里它把 case1 的 `LED-`/`BAT_2` 两对
        net_input 从 21.2/20.9px 收到 3.4/2.9px，且**没有把任何一对已合格的框推坏**。
      · `inter` —— 与旧框求交（只收不放）。同一次复算里结果与 `sym` 相同。
      · `union` —— 与乘积相同的那次复算里 0 修复，默认不给这一档。
    """
    from stage1.bigic import _iou              # 复用既有 IoU 实现
    from stage1.detect import DEDUP_PX, to_json_box

    H = int(bgr.shape[0])
    thr = dedup_iou()
    mode = str(os.environ.get("STAGE1_SYM16_FIX_MODE", "sym") or "sym").strip().lower()
    jitter_min = float(os.environ.get("STAGE1_SYM16_FIX_MIN", DEDUP_PX) or 0.0)
    dets = raw_dets(bgr, conf, imgsz) if dets is None else dets
    changed, retyped, skip = [], [], {}
    for c in comps:
        bb = c.get("bbox_img")
        if not bb or len(bb) < 4:
            continue
        pt = str(c.get("type") or "").strip().lower()
        best = None
        for d in dets:
            t = SYM2TYPE.get(d["cls"])
            if t is None:
                continue
            v = _iou(bb, d["box_img"])
            if v >= thr and (best is None or v > best[0]):
                best = (v, t, d)
        if best is None:
            skip["no_same_type"] = skip.get("no_same_type", 0) + 1
            continue
        v, t, d = best
        nb = [round(float(x), 2) for x in d["box_img"]]
        # ---- 类型改写（对应 `component.type_mismatch` 那 3 对）----
        # 判据刻意保守：位置已经判同（IoU≥去重阈）、sym16 置信度高于既有件的 score、
        # 目标类型在白名单内。逐件记账，便于审计「改了哪些件」。
        if type_change and t != pt and t in TYPE_TARGETS \
                and float(d["score"]) > float(c.get("score") or 0.0):
            retyped.append({"id": c.get("id"), "from": pt, "to": t,
                            "conf": round(float(d["score"]), 4),
                            "was_score": round(float(c.get("score") or 0.0), 4),
                            "iou": round(v, 3)})
            c["type"] = t
            c["proposal"] = t
            c["sym16_type"] = t
            pt = t
        if t != pt:                            # 改不动的（目标类型不在白名单）就只记不报
            skip["type_blocked"] = skip.get("type_blocked", 0) + 1
            continue
        # ---- 框精修 ----
        if [round(float(x), 2) for x in bb] == nb:
            skip["same_box"] = skip.get("same_box", 0) + 1
            continue
        if mode == "inter":
            ix0, iy0 = max(float(bb[0]), nb[0]), max(float(bb[1]), nb[1])
            ix1, iy1 = min(float(bb[2]), nb[2]), min(float(bb[3]), nb[3])
            if ix1 <= ix0 or iy1 <= iy0:
                skip["empty_inter"] = skip.get("empty_inter", 0) + 1
                continue
            nb = [round(ix0, 2), round(iy0, 2), round(ix1, 2), round(iy1, 2)]
        if nb[2] - nb[0] <= 0 or nb[3] - nb[1] <= 0:
            skip["degenerate"] = skip.get("degenerate", 0) + 1
            continue
        # ---- 位移显著性门 ----
        # **为什么必须有**：管线在 `symbox.grow` 之后还会用框的几何去定引脚归属（实测
        # case4 只把 U3 的左沿挪 1.1px，`pin_1`/`pin_2` 就换了位 ⇒ netH 0.904→0.840、
        # 整页 87.09→82.32，纯亏）。而真正能吃到 `bbox_mismatch` 的那两对（case1 的
        # `LED-`/`BAT_2`）都是 15~21px 的大位移。
        # 阈取 `stage1/detect.py:54` 的 `DEDUP_PX`：那是既有「两个框近到 12px 以内就算
        # 同一个东西」的判据 —— 小于它的位移连对象身份都改不了，更不可能改成对的框。
        # `STAGE1_SYM16_FIX_MIN` 只是把这道门的阈值暴露出来（默认 = 该既有常量）：
        # 设 0 = 关掉门（13 页实测 mean 82.25→82.31、case4 87.09→82.32，噪声级且带长尾）。
        if max(abs(nb[i] - float(bb[i])) for i in range(4)) < jitter_min:
            skip["jitter"] = skip.get("jitter", 0) + 1
            continue
        changed.append({"id": c.get("id"), "type": pt, "cls": d["cls"],
                        "iou": round(v, 3), "conf": round(float(d["score"]), 4),
                        "from": [round(float(x), 2) for x in bb], "to": nb})
        c["bbox_img"] = nb
        c["bbox_json"] = [round(float(x), 2) for x in to_json_box(nb, H)]
        c["sym16_iou"] = round(v, 3)
    return {"on": True, "imgsz": int(imgsz or IMGSZ), "mode": mode,
            "n_dets": len(dets), "n_fix": len(changed), "n_type": len(retyped),
            "fixed": changed, "retyped": retyped, "skipped": skip}
