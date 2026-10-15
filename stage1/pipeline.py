"""第一阶段总装：PNG → 元件几何 + 导线骨架 + 附着关系 + 多假设槽位。

并行策略（为什么是进程池，不是线程池）
--------------------------------------
用户建议「多线程处理图片」。这里实测后的选择是**进程池**：
- 导线骨架（`pipeline/pskel.py`）是纯 Python 循环 + numpy，**GIL 绑死**，线程池
  给不了加速；
- YOLO 推理虽然释放在 C++，但一个 `YOLO` 实例并发 forward 并不安全；
- 进程池每个 worker 各持一份模型与一份解释器状态，互不干扰，是唯一稳的做法。
**但**：并行只解决「总墙钟」，不解决「单张超时」——单张最大那张骨架 88.5 s 是
硬成本，见 `wires.extract_wires` 的降采样与 `docs/官方口径评测报告.md`。

线程池仍然用在**该用的地方**：`llm.py` 里同时派发多个 LLM 请求（网络 IO 等待）。

输出（每个 case 一个 json）
---------------------------
`{"case", "image", "components", "wires", "attachments", "hypotheses", "timing"}`
"""

from __future__ import annotations


import copy

import json

import math

import os

import sys

import time

import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path

try:                                    # 作为包导入（python -m stage1 / submission\run.py）
    from envflag import env_on      # 「假值词表」唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage1\pipeline.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on


import cv2

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from stage1 import wires as wires_mod          # noqa: E402

from stage1 import pintrace as ray_mod         # noqa: E402

from stage1 import nets as nets_mod            # noqa: E402

from stage1 import pincol as pincol_mod        # noqa: E402

from stage1 import pin_evidence as evidence_mod  # noqa: E402

from stage1 import pincolor as color_mod       # noqa: E402

from stage1 import passive_pins as passive_mod  # noqa: E402

from stage1.detect import Detector             # noqa: E402

from stage1 import detect as det_mod           # noqa: E402


DET = None   # 每个进程一份


# 大 IC 几何补检（`stage1/bigic.py`）。默认 **on**：YOLO 对 `box` 类的召回不够
# （220 张里 960 个 GT IC 框有 446 个整颗没检出），而一颗 IC 的键配不上就带走它
# 全部引脚 —— 引脚权重 0.25，一张图直接从 25 分掉到 0。实测代价 0.02 s/张。
BIGIC_ON = env_on("STAGE1_BIGIC", "1")


# ★★ 引脚**列枚举**（`stage1/pincol.py`）。默认 **on**。
# 为什么必须有：`pins.build_pins` 的主判据是「上游导线触到框边」，所以**没接线
# 的引脚一个都看不见**；官方 `pin_f1` 占 25%，220 张实测只拿到 0.92 分
# （P 0.1111 R 0.0220 F1 0.0367），而 GT 的 box 引脚有 9356 个。
# 而引脚号其实**印在图上**（`work\preview\_jlcbox.png`：0143 的 U1 框外贴着边印
# `1..64`、框内印引脚名）。实图实测同一颗 U1：
#     枚举前 5px 命中   3/64   （stage1 只给 8 个点）
#     枚举后 5px 命中  63/64   （L 32 个 pitch 9.99、R 31 个 pitch 9.96）
PINC_ON = env_on("STAGE1_PINC", "1")


# ★★ 彩色引脚端点实测（队友 `stage1/pincolor.py`）。**默认 off，等 A/B**。
# 为什么需要它：stage2 那五个「引脚车道裁片 / 读数」读取器（`pin_lane_crops` /
# `pin_lane_reading` / `pin_text_crops` / `color_crops` / `pin_number_reading`）
# 的输入是「**已确认的彩色引脚车道**」+ `doc["pin_evidence"]`。我们原来不产这两个
# 键 ⇒ 挂上读取器后实测**零改动**（`work/_tm_ab.ps1`：`tmoff` 71.65 vs `tmon` 71.65，
# 逐案逐通道逐位相同）。要让它生效，必须连同这条证据链一起开。
#   `extract_color_pins` = 只在图像上量「红色桩 + 真实绿线转折」，**不按固定距离
#   摆引脚**（队友的硬约束：短而未连的引脚是合法的；绿色 no-connect 叉不算导线）。
#   `STAGE1_PIN_DIRECTION_GUARD` / `STAGE1_PIN_ATTACHMENT_GUARD` 是队友另两条
#   方向/穿体判据，他们自己默认也是 0（未定案），这里照抄默认值。
PIN_COLOR_ON = env_on("STAGE1_PIN_COLOR", "1")

NON_IC_PINS_ON = env_on("STAGE1_NON_IC_PINS", "1")

MONO_FRAME_PINS_ON = env_on("STAGE1_MONO_FRAME_PINS", "1")


# Direction and body-crossing checks are opt-in until a held-out end-to-end
# comparison confirms they help for the detector's boxes and this drawing style.
PIN_DIRECTION_GUARD = env_on("STAGE1_PIN_DIRECTION_GUARD", "0")

PIN_ATTACHMENT_GUARD = env_on("STAGE1_PIN_ATTACHMENT_GUARD", "0")


def apply_color_pins(components, pins, pin_synth, color_records):
    """Apply measured color terminals after grid merging, before net indexing.

    A confirmed red frame with pins on it supplies stronger evidence than
    gray text/grid guesses. Replace legacy guesses on color-confirmed sides.
    On a fully confirmed red frame all sides are scanned, including sides with
    no red pin strokes. Unconfirmed sides and monochrome components keep the
    legacy candidates. The dropped guesses remain in the diagnostic ledger.

    ★ 逐字并入队友实现（`codex/improve-pin-network-parsing:stage1/pipeline.py:76`），
    只把 `math` 的用法原样保留 —— 被替换掉的旧猜测进 `ledger`，**不静默消失**。
    """
    result = {cid: list(points) for cid, points in pins.items()}
    synth = {cid: list(flags) for cid, flags in (pin_synth or {}).items()}
    overlays, connected, ledger = {}, {}, {}
    for comp in components:
        cid = comp["id"]
        if comp.get("type") != "box":
            continue
        recs = color_records.get(cid) or []
        if not recs:
            continue
        # Do not treat arbitrary caller metadata as a confirmed color stroke.
        recs = [dict(r) for r in recs if r.get("frame_confirmed") is True
                and r.get("edge") in ("top", "right", "bottom", "left")
                and len(r.get("point_img") or []) == 2
                and all(math.isfinite(float(v)) for v in r["point_img"])]
        if not recs:
            continue
        confirmed = {e for r in recs for e in r.get("confirmed_edges", [])}
        covered = {r["edge"] for r in recs}
        if confirmed == {"top", "right", "bottom", "left"} and not PIN_COLOR_KEEP_MISS:
            covered = confirmed
        old = result.get(cid) or []
        old_flags = synth.get(cid) or []
        kept, kept_flags, dropped = [], [], []
        for index, p in enumerate(old):
            edge = evidence_mod._EDGES[evidence_mod._candidate_edge(p, comp["bbox_img"])]
            if edge in covered:
                dropped.append({"index": index, "point_img": list(p), "edge": edge,
                                "reason": "replaced_by_frame_attached_color_strokes"})
            else:
                kept.append(tuple(p))
                kept_flags.append(old_flags[index] if index < len(old_flags) else 0)
        measured = []
        flags, connections = [], []
        for r in recs:
            r["point_img"] = [round(float(v), 1) for v in r["point_img"]]
            if r.get("anchor_img"):
                r["anchor_img"] = [round(float(v), 1) for v in r["anchor_img"]]
            # ★ `STAGE1_PIN_COLOR_CAP`（默认 0 = 逐位老行为）：把实测点沿
            #   「框边锚点 anchor_img → 实测点」方向拉回到距锚点不超过 CAP 像素处。
            #   动机（§5.116）：`pincolor` 的实测点落在红桩**末端**（官方 10GT 外距
            #   中位 19.3px），而官方 GT 的引脚点外距中位 12.0px（均值 12.5，6~22），
            #   `pin_tol=5` ⇒ 整条边被判 position_mismatch：本树该臂 65.14（引脚 0.0953），
            #   队友树同臂 49.39（引脚 0.1332，7 张爆 0）。队友树加 CAP=12 后
            #   57.72/引脚 0.4891（+8.33，见 §5.116），故这里也留同一个可 A/B 的闸。
            if PIN_COLOR_CAP > 0:
                _a = r.get("anchor_img")
                _p = r["point_img"]
                if isinstance(_a, (list, tuple)) and len(_a) == 2:
                    _dx, _dy = _p[0] - _a[0], _p[1] - _a[1]
                    _d = math.hypot(_dx, _dy)
                    if _d > PIN_COLOR_CAP and _d > 1e-6:
                        r["point_img"] = [round(_a[0] + _dx / _d * PIN_COLOR_CAP, 1),
                                          round(_a[1] + _dy / _d * PIN_COLOR_CAP, 1)]
            r["trust"] = "trusted"
            measured.append(tuple(r["point_img"]))
            connections.append(r.get("connected"))
            flags.append(int(r.get("connected") is False))
        result[cid] = measured + kept
        synth[cid] = flags + kept_flags
        overlays[cid] = recs + [None] * len(kept)
        connected[cid] = connections + [None] * len(kept)
        ledger[cid] = {"measured": recs, "replaced_guesses": dropped,
                       "confirmed_edges": sorted(confirmed),
                       "retained_unconfirmed_side": len(kept)}
    return result, synth, overlays, connected, ledger


def overlay_color_evidence(evidence, overlays):
    """Keep evidence exactly aligned with the post-color pin sequence.

    ★ 逐字并入队友实现（`...:stage1/pipeline.py:137`）。
    """
    result = dict(evidence)
    for cid, recs in overlays.items():
        old = result.get(cid) or [{} for _ in recs]
        if len(old) != len(recs):
            old = [{} for _ in recs]
        result[cid] = [dict(r) if r is not None else dict(prior)
                       for prior, r in zip(old, recs)]
    return result


def apply_mono_frame_pins(components, pins, bgr, ink, texts,
                         pin_synth, color_overlays, pin_connected):
    """Recover measured black stroke rows before assigning network indices.

    Old indices are immutable. Colored sides are protected, and estimated
    black endpoints use the tight synthetic-pin network radius. The audit is
    diagnostic only; it never claims to measure a black pin/wire boundary.
    """
    skipped = {}
    for cid, records in color_overlays.items():
        sides = set()
        for rec in records:
            if rec and rec.get("frame_confirmed") is True:
                sides.add(rec["edge"])
                sides.update(rec.get("confirmed_edges") or [])
        if sides:
            skipped[cid] = sides
    # Pass copies so a recovery failure cannot corrupt existing geometry.
    original = copy.deepcopy(pins)
    recovered, index_map, audit = evidence_mod.recover_frame_pins(
        components, original, bgr, ink=ink, texts=texts, skip_sides=skipped)
    if set(recovered) != set(pins):
        raise ValueError("mono recovery changed the component index set")
    for cid, old in pins.items():
        points = recovered[cid]
        if (len(points) < len(old)
                or index_map.get(cid) != {i: i for i in range(len(old))}
                or any(len(p) != 2 or not all(math.isfinite(float(v)) for v in p)
                       for p in points)):
            raise ValueError("mono recovery changed old pin indices or returned invalid points")
        records = color_overlays.get(cid) or []
        if any(rec is not None and tuple(points[i]) != tuple(old[i])
               for i, rec in enumerate(records)):
            raise ValueError("mono recovery altered a measured color contact")
    synth = {cid: list(flags) for cid, flags in pin_synth.items()}
    overlays = {cid: list(records) for cid, records in color_overlays.items()}
    connected = {cid: list(flags) for cid, flags in pin_connected.items()}
    for cid, log in audit.items():
        changes = (log.get("added") or []) + (log.get("calibrated") or [])
        if not changes:
            continue
        count = len(recovered[cid])
        flags = synth.get(cid, [])
        flags += [0] * (count - len(flags))
        for rec in changes:
            flags[rec["index"]] = 1
        synth[cid] = flags
        if cid in overlays:
            overlays[cid] += [None] * (count - len(overlays[cid]))
        if cid in connected:
            connected[cid] += [None] * (count - len(connected[cid]))
        log["old_index_map"] = index_map[cid]
    return recovered, synth, overlays, connected, audit


def overlay_mono_evidence(evidence, audit, pins):
    """Retain the exact measured stroke row instead of searching neighbours again."""
    result = copy.deepcopy(evidence)
    for cid, log in audit.items():
        records = result.get(cid) or [{} for _ in pins.get(cid, [])]
        if len(records) != len(pins.get(cid, [])):
            raise ValueError("mono evidence does not align with final pin indices")
        for change in (log.get("added") or []) + (log.get("calibrated") or []):
            index = change["index"]
            measured = copy.deepcopy(change["evidence"])
            if (not 0 <= index < len(records)
                    or math.dist(measured["point_img"], pins[cid][index]) > 1e-4
                    or measured.get("trust") != "trusted"):
                raise ValueError("mono evidence is stale or lacks a measured stroke")
            measured["estimated_endpoint"] = True
            records[index] = measured
        for rec in records:
            if rec.get("source") == "frame_hollow_marker":
                continue
            for row in log.get("marker_rows") or []:
                marker=row["marker"]
                point=rec.get("point_img") or []
                if (len(point)==2 and rec.get("edge")==row["edge"]
                        and abs(point[1]-row["tangent_center"]) <= (marker["hole_bbox"][3]-marker["hole_bbox"][1])/2+3):
                    # Triangle slopes can look like short straight strokes at
                    # the wrong row. Only the independently measured outer
                    # line defines the row here; preserve the old index/point.
                    rec["trust"]="uncertain"
                    rec["reason"]="hollow_marker_row_requires_outer_line_anchor"
                    rec["marker_row_anchor_img"]=list(row["anchor_img"])
                    break
            if rec.get("trust") != "trusted":
                continue
            strokes=[r for r in log.get("stroke_rows") or [] if r["edge"]==rec.get("edge")]
            point=rec.get("point_img") or []
            if not strokes or len(point)!=2:
                continue
            tangent=point[1] if rec["edge"] in ("left","right") else point[0]
            gaps=[max(r["tangent_band"][0]-tangent,tangent-r["tangent_band"][1],0) for r in strokes]
            if 3<min(gaps)<=12:
                rec["trust"]="uncertain"
                rec["reason"]="candidate_outside_nearby_measured_normal_stroke_band"
                rec["nearby_normal_stroke_gap_pixels"]=float(min(gaps))
        result[cid] = records
    return result


def _pt_seg_dist(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def attach(components, wires, max_dist: float = 24.0, top_k: int = 3):
    """每个元件找最近的导线端点 / 线段。

    「导线的末端必然有元器件」是用户提的一致性先验，这里把它做成可量的数据：
    `endpoint` 命中意味着元件框边上正好有一根线的头；`segment` 命中意味着元件压在
    一根线中间（多半是串联元件）。两者都没有 = 这个元件不接任何线，可疑。
    """
    eps = wires.get("endpoints") or []
    segs = wires.get("segments") or []
    out = []
    for c in components:
        cx, cy = c["center_img"]
        x1, y1, x2, y2 = c["bbox_img"]
        hits = []
        # 端点：算到「框边」的距离（点到矩形的距离），不是到中心
        dx = max(x1 - cx, 0.0, cx - x2)
        dy = max(y1 - cy, 0.0, cy - y2)
        for i, e in enumerate(eps):
            d = math.hypot(max(x1 - e[0], 0.0, e[0] - x2),
                           max(y1 - e[1], 0.0, e[1] - y2))
            if d <= max_dist:
                hits.append({"kind": "endpoint", "index": i, "dist": round(d, 2),
                             "at": [round(e[0], 1), round(e[1], 1)]})
        for i, s in enumerate(segs):
            d = _pt_seg_dist((cx, cy), s[0], s[1])
            if d <= max_dist:
                hits.append({"kind": "segment", "index": i, "dist": round(d, 2),
                             "at": [[round(s[0][0], 1), round(s[0][1], 1)],
                                    [round(s[1][0], 1), round(s[1][1], 1)]]})
        hits.sort(key=lambda h: h["dist"])
        out.append({"id": c["id"], "type": c["type"],
                    "n_endpoint": sum(1 for h in hits if h["kind"] == "endpoint"),
                    "n_segment": sum(1 for h in hits if h["kind"] == "segment"),
                    "hits": hits[:top_k]})
    return out


# ---------------------------------------------------------------- 单 case
def run_case(case, weights: str, imgsz: int = 1024, conf: float = 0.25,
             device: str = "0", max_pixels: int = wires_mod.DEFAULT_MAX_PIXELS,
             tiler=None, use_cv: bool = True, do_wires: bool = True,
             do_text: bool = True, do_rays: bool = True, do_pins: bool = True,
             det: Detector | None = None):
    global DET
    t0 = time.time()
    # ★ 2026-07 审计项②：这张图里「被吞掉的异常」都记到这里，最后写进
    #   `doc["errors"]`。以前 `except Exception: traceback.print_exc()` 之后继续跑，
    #   调用方只看得到 `ok: True` —— 而 pin/netH/net_line 合计 **0.70 权重**可以被
    #   静默清零（`build_pins`/`build_nets` 抛异常时 comp_pins/nets 直接变空，
    #   端到端只会表现为「这张图这些项是 0」，与「这张图本来就没有引脚」不可区分）。
    errs: list[str] = []
    # ★★ 「不删除，只削弱」账本（m40034 第四项）：**每个 case 清零**（ProcessPool 的
    #   worker 会被复用，不清就会把上一张图的事件带进这一张）。
    try:
        from cvparts.hypo import reset_ledger
        reset_ledger(case.case_id)
    except Exception:
        pass
    bgr = _imread_bgr(case.png_path)
    if bgr is None:
        raise RuntimeError(f"读不出图: {case.png_path}")
    H, W = bgr.shape[:2]
    # ★ 2026-07 审计项④：`DET` 是模块级缓存，同进程跑 220 张时从第二张起
    #   `weights` / `imgsz` / `device` 这三个参数**被静默忽略**（只有 `conf` 每次
    #   显式传进 `detect()`）—— 于是「先用默认权重跑几张、再换权重跑其余」会得到
    #   一张张配置混杂的结果而没有任何提示。这里改成：配置对不上就重建并打印一行。
    #   批量调用时每个 case 传的 kw 相同 ⇒ 只在第一张建一次，行为与以前一致。
    if det is None and DET is not None:
        _mism = [f"{k} {getattr(DET, k)} -> {v}"
                 for k, v in (("weights", str(weights)), ("imgsz", imgsz),
                              ("device", device))
                 if str(getattr(DET, k, None)) != str(v)]
        if _mism:
            print(f"[stage1] 缓存的检测器配置与本次请求不同，重建：{'；'.join(_mism)}",
                  flush=True)
            DET = None
    d = det or DET or Detector(weights, imgsz=imgsz, conf=conf, device=device)
    DET = d

    # ★ 独家归属的账（业主 m25181：「最终只会归属于某一个……其他地方同理」）。
    #   由 `detect()` 逐条写入：谁、输给谁、差多少、按哪条规则。
    own_ledger: list = []
    comps, hyps, tm, cnt = d.detect(bgr, conf=conf, use_cv=use_cv, tiler=tiler,
                                    own_ledger=own_ledger)
    hyps.case = case.case_id

    # 大 IC 几何补检：必须在 `extract_wires`（吃 `occlude=[c["bbox_img"] ...]`）和
    # `pintrace.trace_from_boxes`（吃 `cboxes`）**之前**加进来，这样 IC 边框会被
    # 涂掉不污染导线骨架，而且大 IC 的四条边会自己长出引脚射线。
    t_big, n_big = 0.0, 0
    if BIGIC_ON:
        try:
            from stage1 import bigic as bigic_mod
            _t = time.time()
            n_big = bigic_mod.add_components(bgr, comps, hyps)
            t_big = time.time() - _t
        except Exception as e:
            traceback.print_exc()
            errs.append(f"bigic: {type(e).__name__}: {e}")
            n_big = 0

    # 文字**定位**属第一阶段（用户 m03802 第三条指令），内容识别才属第二阶段。
    # 两个用途：①涂掉再跑骨架 —— 文字是导线假阳性的最大来源；②给导线端点当锚点，
    # 没有它「悬空导线」判不出来（jlc 风格靠 net label 连网，只有元件框一个锚点时
    # 全部真导线都会被误判成悬空：实测悬空元件 75 → 1、suspect 占比 20.5% → 3.9%）。
    t_txt = 0.0
    if do_text:
        from cvparts.textbox import detect_boxes as _tb
        t1 = time.time()
        texts = _tb(bgr)
        t_txt = time.time() - t1
    else:
        texts = []
    w = {"n_segments": 0, "n_endpoints": 0, "n_junctions": 0, "segments": [],
         "endpoints": [], "junctions": [], "scale": 1.0, "n_polylines": 0,
         "n_thin_px": 0, "has_chroma": False, "n_nodes": 0, "n_branch_paths": 0}
    t_w = 0.0
    t_r = 0.0
    rays: list = []
    if do_wires:
        # 关键：元件框已经检出，跑骨架前先把它们涂掉（联合解算）。
        # 骨架化对墨迹一视同仁，IC 边框/电容极板/引脚名文字全是细笔画 —— 实测不遮挡时
        # 提取总长是 GT 网表的 5.04 倍、只有 6% 压在真导线上。遮挡后 jlc 的 P 从
        # 0.049 升到 0.154、F1 从 0.084 升到 0.191；同时去掉图纸边框再省一半长度。
        # 注意 min_len 必须留 0：pskel 的 snap 会把长走线切成一堆短段，在 segment 级
        # （以及 polyline 级）做长度过滤都会把真导线一起删掉（0101 上 F1 直接归零）。
        #
        # max_axis_dev 则相反 —— 它是唯一有效的形状先验：原理图走线只走 0/45/90/135 度，
        # 而剩下最多的假阳性是文字笔画（任意角度）。6 个代表 case 上 8 度门槛 5 升 1 微降，
        # 且精度普遍上升、召回基本不动：0101 0.191→0.230、0050 0.178→0.236、
        # 0059 0.169→0.202、0107 0.088→0.100、0011 0.028→0.042；只有 KiCad 0001
        # 0.168→0.151（因为它的真导线有不少 45 度斜段，被 8 度门槛误伤）。
        # 取 12 度是两者之间更稳的折中：0101 0.223 / 0001 0.158。
        # 大尺度连通过滤（用户的核心洞察）：文字笔画是孤立的小团（extent 中位数 10px、
        # p90 18px），导线是一个大尺度连通体。实测它能砍掉大量文字假阳性 ——
        # 0059 Datasheet F1 0.193→0.343、0050 Altium 0.216→0.304。
        # 但对 jlc 是净损失（0101 0.223→0.187）：那种风格靠 net label 表达连通，
        # GT 的边本来就是「引脚到标签」的短桩，本来就在小分量里。
        # rescue_pad 修掉这个损失（用户第二句话：导线是用来连接元器件的）：分量再小，
        # 只要端点贴着已检出元件框就仍然保留。ext>=80 + rescue 8 在 6 个 case 上
        # **全部优于**不设门槛的基线（0101 0.223→0.252、0107 0.096→0.111、
        # 0001 0.158→0.165、0059 0.193→0.315、0050 0.216→0.295、0011 0.036→0.049）。
        # ★ 2026-10-10（用户 m42009「导线等元件、元件等导线」）：**涂白口径可选**。
        #   `all`（默认 = 历史行为）= 涂掉所有已检出元件框；
        #   `keep` = 一根都不涂；`big` = 只涂大框（min side >= `STAGE1_OCC_BIG`，默认 60）。
        #   28 张导线级实测（`work\tools\dev\_occpol.py`）：不涂时 GT 边覆盖 1093→1240
        #   （**+147 条，R 0.2887→0.3275**），但假阳段多 1615 条（P 0.3577→0.2629），
        #   F1 反而 0.3195→0.2917 ⇒ 端到端要不要换口径必须 A/B 判，不能凭通道读数定。
        #   框内证据确实死在我们自己的框上；上游三道闸门已排除（`_prunepol.py`：框内覆盖
        #   恒为 17/662，`min_extent`/`max_axis_dev`/`_is_frame` 一条框内边都没吃掉）。
        _occ_mode = os.environ.get("STAGE1_OCC_MODE", "all").strip().lower()
        _occ_boxes = [c["bbox_img"] for c in comps]
        if _occ_mode in ("keep", "none", "off", "0", ""):
            _occ_boxes = []
        elif _occ_mode == "big":
            try:
                _occ_big = float(os.environ.get("STAGE1_OCC_BIG", "60"))
            except ValueError:
                _occ_big = 60.0
            _occ_boxes = [b for b in _occ_boxes
                          if min(float(b[2]) - float(b[0]), float(b[3]) - float(b[1])) >= _occ_big]
        w, t_w = wires_mod.extract_wires(
            bgr, max_pixels=max_pixels,
            occlude=_occ_boxes, erase=texts, erase_pad=1.0,
            drop_frame=True,
            max_axis_dev=12.0, min_extent=80.0, rescue_pad=8.0)

        # ---- 元件引脚介导的贪心射线（用户 m04178 的思路）----
        # 全局骨架的判据天生是「长/直/大连通」，于是三类真导线被丢掉：被文字涂断的、
        # 被元件遮挡的、本来就在小分量里的（jlc 靠 net label 连网，GT 的边就是「引脚到
        # 标签」的短桩）。但导线一定从引脚出发，方向由引脚给定 —— 所以反过来从引脚
        # 沿法线外推，一步一像素跟着墨走，直到墨断掉。
        # 关键：射线跑在**原始墨迹**上，不涂文字、不遮挡元件 —— 方向已知，文字带不偏它。
        # 实测（官方 220 张，tol=5px，判据=GT 段上每点都要有预测段）：
        #   骨架 F1 0.2701 / R 0.2730     射线 F1 0.2661 / R 0.5459（总长比只 0.66）
        #   并集 F1 0.3569 / R 0.6827  ⇒ 召回 2.5 倍、F1 +32%
        # 归因：骨架漏掉的 GT 边里，射线救回 `other` 的 64%、`in_text` 的 54%。
        # **锚点截断（用户 m04360/m04510，已实测为本项目最优配置）**：射线走到
        # 「碰到第一个锚点」就截断，锚点 = 别的元件框 + 文字框（网络标签）；走到尽头
        # 也没碰到任何锚点的整条丢掉。上一次的证伪是操作化错误：走到底的射线早就冲过了
        # 标签，远端当然不在锚点旁。60 张同尺对照（连通性 = 同 net 引脚对是否同分量）：
        #   骨架 only           连通 F1 0.4939（R 0.386 / P 0.686）
        #   骨架+射线（旧）      连通 F1 0.4080（R 0.735 / P 0.282）
        #   骨架+射线+锚点截断   连通 F1 0.5421（R 0.590 / P 0.502）  ← 最优
        # 按官方权重（net_hypergraph 0.35 + net_line 0.10）折算：0.194 / 0.187 / 0.224。
        if do_rays and do_wires:
            t1 = time.time()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ink = (gray < ray_mod.INK).astype(np.uint8)
            cboxes = [c["bbox_img"] for c in comps]
            # **不要加 anchors / clip_anchors / require_anchor**：修好连通性评测器之后
            # 重测，锚点截断在两个尺子上都是净亏 ——
            #   臂                覆盖 F1   连通 R   连通 F1
            #   full+rays         0.4434   0.1389   0.2439
            #   full+rays-clip    0.3536   0.0813   0.1504
            #   full+rays-anchor  0.3394   0.0678   0.1270
            # 根因：`require_anchor=True` 会把「走到尽头没碰到锚点」的射线整条丢掉，
            # 引脚未绑定率从 14.9% 抬到 26.6%，连通召回反而腰斩。
            rays = ray_mod.trace_from_boxes(ink, cboxes)
            t_r = time.time() - t1
            if rays:
                w["segments"] = list(w.get("segments", [])) + rays
    att = attach(comps, w) if do_wires else []

    # ---- 引脚 + 网络：stage1 之前完全没有这两样，pin 0.25 + net 0.35 + line 0.10
    # = 0.70 分在结构上拿不到。这里补上「段 -> 连通分量 -> 挂引脚 -> net」这一步。
    comp_pins, nets, seg_groups, pin_s = {}, [], [], 0.0
    pincol_cols, pincol_stat, pin_synth = {}, {}, {}
    # ★ 队友证据链的产物（默认空 ⇒ 下游一切如旧）：`pin_evidence` = 逐脚图像证据，
    #   `pin_color` = 被彩色实测替换后的引脚序列（进 `pin_evidence` 的对齐用），
    #   `color_overlays` = 与最终引脚序列**逐位对齐**的彩色记录（含 None 占位）。
    pin_evidence = {}
    color_overlays, pin_connected, pin_color = {}, {}, {}
    passive_pin_report = {}
    non_ic_pin_structure = {}
    symbol_groups = []
    pin_mono = {}
    # ★★ 「不丢弃」的**留痕**通道（owner 哲学：证据不能被悄悄抹掉）。
    #   `build_nets` 会把「够不着任何导线而被硬截断的引脚」写进 `orphans`、
    #   把「软判据区间内勉强挂上的引脚」写进 `weak`。历史行为是 `if g < 0: continue`
    #   —— 那些引脚**在下游完全不存在**，于是既没接线、又不知道曾经有过。
    #   现在它们进 `doc["pin_join"]`，`stage2` / 复核轮 / 可视化都能看见。
    net_report = {}
    if do_wires and do_pins:
        t_p = time.time()
        try:
            # ★ 墨迹掩膜同时喂给「引脚提取」：`STAGE1_PIN_STUB=1` 时用它补几何引脚桩，
            # 因为「导线触到框边」这条主判据看不见**没接线**的引脚（220 张实测
            # 覆盖率只有 0.45、`point@5px` 仅 16.8%）。
            _ink = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < ray_mod.INK).astype(np.uint8)
            comp_pins = nets_mod.build_pins(
                comps, w.get("segments") or [], texts=texts, ink=_ink,
                direction_guard=PIN_DIRECTION_GUARD)
            # ★★ 引脚**列枚举**：把「框外那一排引脚号 / 框内那一排引脚名」的
            # 等距格点补进 `comp_pins`。规则是**不丢弃**：枚举点附近已有导线触点
            # 就用触点（网络装配靠它），没有触点就用格点坐标（= 补出来的脚），
            # 剩下没被认领的触点原样保留。详见 `stage1\pincol.py` 的模块注释。
            if PINC_ON:
                try:
                    pincol_cols = pincol_mod.enumerate_pins(
                        bgr, comps, texts, ink=_ink)
                    comp_pins, pin_synth, pincol_stat = pincol_mod.merge_into(
                        comp_pins, pincol_cols)
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"pincol: {type(e).__name__}: {e}")
            # ★★ 彩色引脚端点实测（队友 `stage1/pincolor.py`，默认关）：
            #   在「已确认的红色框」上把灰色/格点猜出来的旧引脚换成**实测端点**。
            #   只动「彩色已确认的那几侧」，未确认侧与黑白元件保持原样；被换掉的
            #   旧猜测进 `ledger`（诊断账本），不会静默消失。
            if PIN_COLOR_ON:
                try:
                    color_records = color_mod.extract_color_pins(bgr, comps)
                    comp_pins, pin_synth, color_overlays, pin_connected, pin_color = \
                        apply_color_pins(comps, comp_pins, pin_synth, color_records)
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"pin_color: {type(e).__name__}: {e}")
            if MONO_FRAME_PINS_ON:
                try:
                    comp_pins, pin_synth, color_overlays, pin_connected, pin_mono = apply_mono_frame_pins(
                        comps, comp_pins, bgr, _ink, texts,
                        pin_synth, color_overlays, pin_connected)
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"pin_mono: {type(e).__name__}: {e}")
            if NON_IC_PINS_ON:
                try:
                    comp_pins, passive_pin_report = passive_mod.refine_pins(
                        comps, comp_pins, bgr)
                    # Terminal replacement invalidates index-based flags from
                    # the old sequence. IC candidates and their flags stay aligned.
                    for cid in passive_pin_report:
                        pin_synth.pop(cid, None)
                        pin_connected.pop(cid, None)
                    from stage1 import rgb_led
                    comps, comp_pins, passive_pin_report, symbol_groups = rgb_led.group_components(
                        comps, comp_pins, passive_pin_report, bgr)
                    for group in symbol_groups:
                        group['original_attachments'] = [a for a in att if a.get('id') in group['member_ids']]
                        for member in group['member_ids']:
                            pin_synth.pop(member, None)
                            pin_connected.pop(member, None)
                            pin_evidence.pop(member, None)
                    if symbol_groups:
                        # Raw LED sub-boxes left the package's common red
                        # structure in the skeleton/rays. Mask the whole
                        # recovered package before rebuilding external wires,
                        # retaining already computed IC terminal candidates.
                        w, _rgb_wire_s = wires_mod.extract_wires(
                            bgr, max_pixels=max_pixels,
                            occlude=[c['bbox_img'] for c in comps], erase=texts, erase_pad=1.0,
                            drop_frame=True, max_axis_dev=12.0, min_extent=80.0, rescue_pad=8.0)
                        t_w += _rgb_wire_s
                        rays = []
                        if do_rays:
                            _rgb_ray_start = time.time()
                            rays = ray_mod.trace_from_boxes(_ink, [c['bbox_img'] for c in comps])
                            rays = rgb_led.clip_external_rays(rays, symbol_groups)
                            t_r += time.time() - _rgb_ray_start
                            if rays:
                                w['segments'] = list(w.get('segments', [])) + rays
                        for group in symbol_groups:
                            group['external_wires_rebuilt'] = True
                        att = attach(comps, w)
                    from stage1 import non_ic_pin_structure as non_ic_structure_mod
                    comp_pins, non_ic_pin_structure = non_ic_structure_mod.reconcile(
                        comps, comp_pins, bgr, passive_pin_report)
                    for cid in non_ic_pin_structure:
                        pin_synth.pop(cid, None)
                        pin_connected.pop(cid, None)
                        added = non_ic_pin_structure[cid].get('new_raw_indices') or []
                        if added:
                            # Count evidence does not establish an exact gray
                            # contact. The recovered legacy estimate receives
                            # the existing conservative synthetic-pin radius.
                            pin_synth[cid] = [i in added for i in range(len(comp_pins[cid]))]
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"passive_pins: {type(e).__name__}: {e}")
            # ★ 把**墨迹掩膜**传给网络装配：`_weld_unions` 用它判断两个端点之间的
            # 直线上是否真的有墨。结点圆点造成的断口是**画出来的**（圆点本身就是墨），
            # 而「把不相干的两条线凑近」的假配对连线必然穿过空白。
            # 有了这道判据，`STAGE1_WIRE_WELD=24 / STAGE1_WIRE_PERP=14` 才可能
            # 在 10GT 上赚的同时不在 220 张上亏（见 `stage1\labels.py` 的注释表）。
            nets, seg_groups = nets_mod.build_nets(
                comp_pins, w.get("segments") or [], ink=_ink,
                join_far=nets_mod.PIN_JOIN_FAR, report=net_report,
                # ★★ 「并行导线等距对齐」：`pincol` 标出来的**悬空补脚**改用
                # 紧半径（`PIN_JOIN_SYNTH=3`）挑导线。引脚列间距只有 10px，
                # 而 `PIN_JOIN_DIST=12` 比它还大 ⇒ 补脚会认领**隔壁脚的导线**，
                # 把两条网络并成一条。`pin_synth` 为空时行为与历史完全一致。
                pin_synth=pin_synth, components=comps,
                guard_attachments=PIN_ATTACHMENT_GUARD, pin_connected=pin_connected,
                # ★★ 「抹字断口」定向桥接：`texts` 就是上面 `extract_wires(...,
                # erase=texts)` 抹掉的那批文本框（`STAGE1_TEXT_BRIDGE>0` 才生效）。
                # 只认「两个线端各贴住同一个被擦除文本框」的缺口，绝不认元件框 ——
                # 抬全局容差会把电容两块极板并成一坨（见 `stage1\labels.py` 注释）。
                bridges=([(t.get("bbox_img") if isinstance(t, dict) else t)
                          for t in (texts or [])
                          if (t.get("bbox_img") if isinstance(t, dict) else t)] or None))
        except Exception as e:
            traceback.print_exc()
            # ★ 审计项②：这一条是**最贵**的吞异常 —— 它把 pin(0.25)+net(0.35)
            #   +line(0.10) = 0.70 权重整块清成 0，而这一层以前只 print 一下就继续。
            errs.append(f"pins/nets: {type(e).__name__}: {e}")
            comp_pins, nets, seg_groups = {}, [], []
        pin_s = round(time.time() - t_p, 2)

        # ★★ 逐脚图像证据（队友 `stage1/pin_evidence.py`，默认算，成本 ~0.1 s/张）：
        #   它必须对着**最终合并后**的引脚序列算 —— 坐标在这一行之前已经定稿。
        #   它的失败**不能**抹掉本来有效的引脚/网络（所以自己一层 try）。
        #   `ink` 用二维墨迹掩膜（`_ink`），`texts` 只当**软证据**（文字掩膜），
        #   绝不擦像素；最后用 `overlay_color_evidence` 把彩色那批记录对齐回来。
        try:
            pin_evidence = evidence_mod.build_pin_evidence(
                comps, comp_pins, _ink, texts)
        except Exception as e:
            traceback.print_exc()
            errs.append(f"pin_evidence: {type(e).__name__}: {e}")
            pin_evidence = {}
        # ★★ 黑白框实测校准留下的那行证据直接复用（队友 `overlay_mono_evidence`）：
        #   顺序与队友一致（先 mono 后 color）。`pin_mono` 为空时**一次都不调用**，
        #   保证 `STAGE1_MONO_FRAME_PINS=0` 的路径与历史逐位相同（无 deepcopy 开销）。
        if pin_mono:
            try:
                pin_evidence = overlay_mono_evidence(pin_evidence, pin_mono, comp_pins)
            except Exception as e:
                traceback.print_exc()
                errs.append(f"pin_mono_overlay: {type(e).__name__}: {e}")
        pin_evidence = overlay_color_evidence(pin_evidence, color_overlays)

    tm = dict(tm)
    tm["wires_s"] = round(t_w, 2)
    tm["rays_s"] = round(t_r, 2)
    tm["text_s"] = round(t_txt, 2)
    tm["bigic_s"] = round(t_big, 2)
    tm["pins_s"] = pin_s
    tm["total_s"] = round(time.time() - t0, 2)
    doc = {
        "case": case.case_id,
        "image": {"path": str(case.png_path), "w": W, "h": H},
        "components": comps,
        "pins": {cid: [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pl]
                 for cid, pl in comp_pins.items()},
        # ★ 队友证据链的两个键（`STAGE1_PIN_COLOR` 关时：`pin_color` 为空 dict，
        #   `pin_evidence` 仍是逐脚实测记录 —— stage2 的读取器只认「已确认车道」，
        #   所以证据本身不改变任何输出，只是把「为什么这么判断」留给下游）。
        "pin_evidence": pin_evidence,
        "pin_color": pin_color,
        "passive_pin_report": passive_pin_report,
        "non_ic_pin_structure": non_ic_pin_structure,
        "pin_mono": pin_mono,
        "nets": nets,
        # ★ 引脚接线的「留痕」：`orphans` = 够不着导线而被截断的引脚（历史上直接消失）、
        #   `weak` = 软判据区间内勉强挂上的引脚（带 `d`，下游可降权）。**不参与评分**，
        #   只是让「什么东西被丢掉了」在交付物里可见 —— 这正是 owner 要的「刷新」凭据：
        #   只有先看得见「哪些引脚没锁上」，才谈得上围绕它再扫一次。
        "pin_join": {"n_orphans": len(net_report.get("orphans", [])),
                     "n_weak": len(net_report.get("weak", [])),
                     "n_blocked": len(net_report.get("blocked", [])),
                     "n_ambiguous": len(net_report.get("ambiguous", [])),
                     "direction_guard": PIN_DIRECTION_GUARD,
                     "attachment_guard": PIN_ATTACHMENT_GUARD,
                     "orphans": net_report.get("orphans", []),
                     "weak": net_report.get("weak", []),
                     "blocked": net_report.get("blocked", []),
                     "ambiguous": net_report.get("ambiguous", [])},
        # ★ 独家归属的账（业主 m25181：相邻两个框争一个位置，最终只归一个；
        #   「在其他任何地方，类似的情况也是同理」）。`rule` 是判据（默认
        #   `cd<=12` —— 19 张实测并 2 对、误伤 0），`dropped` 的主张**没有消失**，
        #   它以 `OWN_DEMOTE` 折进了 `kept` 那一槽的候选里（削弱，不覆盖）。
        "ownership": {"rule": det_mod.OWN_RULE, "demote": det_mod.OWN_DEMOTE,
                      "n_dropped": len(own_ledger), "ledger": own_ledger},
        "texts": [{"id": f"t{i:04d}", "bbox_img": [round(float(v), 1) for v in b]}
                  for i, b in enumerate(texts)],
        "wires": {k: v for k, v in w.items() if k not in ("segments", "endpoints",
                                                          "junctions", "polylines")},
        "wire_detail": {"segments": w.get("segments", []),
                        "endpoints": w.get("endpoints", []),
                        "junctions": w.get("junctions", []),
                        "groups": seg_groups,
                        "n_rays": len(rays)},
        "attachments": att,
        # ★★ 引脚**列枚举**的账（`stage1/pincol.py`）：每个 `box`/`bigic` 元件的
        # 引脚记录（图像坐标），带 `side`/`pitch`/`synth` 与两个文字窗
        # （`out_win` 框外号码窗、`in_win` 框内名字窗）以及落在窗里的文字框下标
        # （`out_texts`/`in_texts`）。**阶段二直接用它读号读名**，不用再靠
        # 「谁的 2D 距离最近」去猜 —— 引脚号就在那一个窗里。
        "pincol": pincol_cols,
        "pincol_stat": pincol_stat,
        "hypotheses": hyps.to_dict(),
        # ★★ 「不删除，只削弱」的账本（m40034 第四项）：这一张图里所有「本来会被丢掉」
        #   的东西都在这里留痕 —— `pincol` 整颗 IC 被闸门丢掉、引脚列并入时的补脚与
        #   未认领触点、槽位改写、候选溢出。每条都带**严格为正**的降权分
        #   （0 在很多下游等于「删除」）+ 判据 + 重扫队列。**不参与评分**，
        #   只是让「丢了什么」在交付前的产物里可查。
        "no_delete": _no_delete_dump(),
        "timing": tm,        # ★ 审计项②：非空就说明这张图的某一块被降级跑过（`stats` 里的 0 可能是
        #   「真没有」也可能是「抛异常」，看这里区分）。空列表 = 全链路正常。
        "errors": errs,
        "stats": {"n_components": len(comps), "n_texts": len(texts),
                  "n_pending": sum(1 for c in comps if c["pending"]),
                  "n_attached": sum(1 for a in att if a["n_endpoint"] + a["n_segment"] > 0),
                  "n_pins": sum(len(v) for v in comp_pins.values()),
                  "n_nets": len(nets),
                  "n_pins_in_nets": sum(len(n["pins"]) for n in nets),
                  "n_bigic": n_big,
                  "n_errors": len(errs),
                  "auto_resolve": cnt},
    }
    if symbol_groups:
        doc['symbol_groups'] = symbol_groups
    return doc


# ---------------------------------------------------------------- 批
def _mark(r) -> str:
    """控制台标记：`ok` / `ok*N`（跑通但有 N 处被降级的异常）/ `FAIL`。"""
    if not r.get("ok"):
        return "FAIL"
    n = len(r.get("errors") or [])
    return f"ok*{n}" if n else "ok"


def _worker(job):
    """进程池 worker：job = (case_dict, out_dir, kwargs)。"""
    case, out_dir, kw = job
    # ★ 每个进程各自设显存配额（`set_per_process_memory_fraction` 是按进程的）。
    # 必须在 `run_case` 里任何 CUDA 初始化之前 —— 这里正是最早的位置。
    try:
        from gpu_mem import cap_vram
        cap_vram()
    except Exception:                                               # noqa: BLE001
        pass
    try:
        from schematic_parser.data import Case
        doc = run_case(Case(**case), **kw)
        p = Path(out_dir) / f"{doc['case']}.stage1.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return {"case": doc["case"], "ok": True, "out": str(p),
                "errors": doc.get("errors") or [],
                "timing": doc["timing"], "stats": doc["stats"]}
    except Exception as e:
        return {"case": case.get("case_id", "?"), "ok": False,
                "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-600:]}
    finally:
        # 长跑时把 caching allocator 的碎片还回去，别让它一张张图累积。
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                           # noqa: BLE001
            pass


def run_cases(cases, out_dir, workers: int = 1, **kw):
    """cases: list[Case]。workers>1 走进程池。

    ★ 灾难现场加固（用户 m42543）：进程池的 `f.result()` 以前**没有 try**。
    子进程被 OOM-kill / 段错误时抛的是 `BrokenProcessPool`，它会从 `as_completed`
    冒到调用方（`submission/run.py:233` 也没 try）⇒ **整进程死、`final_results/`
    一份产物都没有、退出码还是 0**（最坏的「整批 0 分」）。现在：
      1. 每个 future 的 `result()` 单独 try，坏掉的那张记 `ok: False` + 原因，其余照跑；
      2. 池级异常（`BrokenProcessPool`/`OSError`）也兜住，**余下没跑完的图在本进程
         串行补跑** —— 慢，但能出分。`cancel_futures=True` 防止 shutdown 时二次抛。
    """
    jobs = [(dict(case_id=c.case_id, split=c.split,
                  png_path=c.png_path, json_path=c.json_path), str(out_dir), kw)
            for c in cases]
    results = []
    done = set()
    if workers <= 1:
        for j in jobs:
            r = _worker(j)
            results.append(r)
            done.add(j[0]["case_id"])
            _report_one(r)
        return results
    ex = ProcessPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(_worker, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            cid = j[0]["case_id"]
            try:
                r = f.result()
            except Exception as e:                                  # noqa: BLE001
                r = {"case": cid, "ok": False,
                     "error": "worker 崩溃（%s: %s）" % (type(e).__name__, e)}
            done.add(cid)
            results.append(r)
            _report_one(r)
    except Exception as e:                                          # noqa: BLE001
        left = [j for j in jobs if j[0]["case_id"] not in done]
        print("[stage1] ⚠️ 进程池整体崩溃（%s: %s）—— 余下 %d 张改在本进程串行补跑"
              % (type(e).__name__, e, len(left)), flush=True)
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:                                           # noqa: BLE001
            pass
    for j in jobs:                          # 串行补跑（池没坏时 left 为空，零开销）
        cid = j[0]["case_id"]
        if cid in done:
            continue
        r = _worker(j)
        done.add(cid)
        results.append(r)
        _report_one(r)
    return results


def _no_delete_dump() -> dict:
    """「不删除，只削弱」账本的紧凑快照（写进 `doc["no_delete"]`）。

    见 `cvparts/hypo.py` 的 `Ledger`：owner m23945 的规矩是「可以削弱内容的置信度，
    但不能轻易覆盖」「超限项必须留在系统里，附一个可比较的标量，由下游降权或复核」
    （`docs/reports/工程哲学.md:41`、`:94`）。这里只是把账本原样交出来给复核层，
    **不参与评分、不改任何判据**。
    """
    try:
        from cvparts.hypo import LEDGER
        return LEDGER.dump()
    except Exception as e:              # 账本永远不该让主链挂掉
        return {"error": "%s: %s" % (type(e).__name__, e)}

# ★ 彩色实测点的**外伸量上限**（px；0 = 关，逐位老行为）。见 §5.116：
#   `pincolor` 实测点取在红桩末端（外距中位 19.3px），官方 GT 约定只有 12.0px
#   （均值 12.5、范围 6~22），而判分器 `pin_tol=5` ⇒ 该臂整边 position_mismatch。
#   队友树实测：CAP=12 让他们的引脚 F1 0.1332→0.4891、总分 49.39→57.72（+8.33）。
PIN_COLOR_CAP = float(os.environ.get("STAGE1_PIN_COLOR_CAP", "0") or 0)

# ★ 四边全 confirm 时，**没有实测记录**的边是否也一并丢掉（队友原语义 = 丢）。
#   动机（§5.119）：新 QA 承认现有 10GT 的引脚坐标不合 PDF 附件三（红绿交接处）并将更新，
#   届时「彩色实测」是候选正解；但 `apply_color_pins` 在四边全 confirm 时把**整件**旧猜测丢掉，
#   是它今天引脚 F1 爆 0 的直接原因。设 1 ⇒ 只丢**真有实测记录**的边，
#   把「测量带来的收益」与「丢掉猜测造成的损失」分开评估（默认 0 = 逐位老行为）。
PIN_COLOR_KEEP_MISS = env_on("STAGE1_PIN_COLOR_KEEP_MISS", "0")


# ---------------------------------------------------------------- 读图（灾难现场加固）
def _max_pixels_hard() -> int:
    """巨图解码护栏阈值（像素数）。`STAGE1_MAX_PIXELS_HARD=0` 关掉这道护栏。"""
    try:
        return int(os.environ.get("STAGE1_MAX_PIXELS_HARD") or 80_000_000)
    except Exception:                                               # noqa: BLE001
        return 80_000_000


def _imread_bgr(path):
    """读成 BGR（`IMREAD_COLOR`）。**比 `cv2.imread` 多两道保险**（用户 m42543）：

    1. **非 ASCII / 带空格路径**：`cv2.imread` 在 Windows 上对非 ASCII 路径会**静默
       返回 `None`**（官方 case 名明确可能带空格），而 `np.fromfile` + `cv2.imdecode`
       不看路径编码。读不出才退回 `imread`（保留旧行为，别把能跑的路堵死）。
    2. **巨图护栏**：像素数超过 `STAGE1_MAX_PIXELS_HARD`（默认 8e7 = 80 MP，只看文件头，
       不解码）时按 1/2、1/4、1/8 解码。硬 OOM 是 `except` 救不回来的死法
       （进程被内核杀掉，整批 0 产物），宁可掉精度也要活着出分。本地最大图
       4491x3173 = 14 MP，远在阈值下 ⇒ **正常路径逐像素不变**。
    """
    p = str(path)
    buf = None
    try:
        buf = np.fromfile(p, dtype=np.uint8)
    except Exception:                                               # noqa: BLE001
        buf = None
    if buf is not None and buf.size:
        flag = cv2.IMREAD_COLOR
        hw = None
        try:
            from PIL import Image
            with Image.open(p) as im:                               # 只读文件头
                hw = (im.height, im.width)
        except Exception:                                           # noqa: BLE001
            hw = None
        if hw is not None:
            lim = _max_pixels_hard()
            pix = int(hw[0]) * int(hw[1])
            if lim > 0 and pix > lim:
                k = 2
                while k < 8 and pix / float(k * k) > lim:
                    k *= 2
                flag = {2: cv2.IMREAD_REDUCED_COLOR_2,
                        4: cv2.IMREAD_REDUCED_COLOR_4}.get(
                            k, cv2.IMREAD_REDUCED_COLOR_8)
                print("[stage1] ⚠️ 巨图 %dx%d = %.0f MP > %.0f MP，按 1/%d 解码"
                      "（硬 OOM 会杀掉整个进程，整批 0 产物）"
                      % (hw[1], hw[0], pix / 1e6, lim / 1e6, k), flush=True)
        img = cv2.imdecode(buf, flag)
        if img is not None:
            return img
    return cv2.imread(p, cv2.IMREAD_COLOR)


def _blame(r):
    """★ 失败时把 error / tb 打出来（评测机踩坑后补的：`FAIL` 后面必须有下文）。

    原先 `run_cases` 只打 `_mark(r)`，`FAIL` 后面就没有下文了，而 `_worker` 早就把
    `error` 和 `tb` 塞进了返回字典（见 `:505`）—— 于是「哪张图、为什么挂」在容器日志里
    完全不可见（第 5 次提交就是 10 张图全是同一个 FAIL，只能从 decision_log.json 反查）。
    这里只加日志，不改任何判定。
    """
    if r.get("ok"):
        return
    print(f"      !! error: {r.get('error')}", flush=True)
    tb = (r.get("tb") or "").rstrip()
    for line in tb.splitlines():
        print(f"      !! {line}", flush=True)


def _report_one(r):
    print(f"  {r['case']:<20}{_mark(r)}  "
          f"{r.get('timing', {}).get('total_s', '?')} s", flush=True)
    _blame(r)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("用 stage1/__main__.py 跑：python -m stage1 --cases 0050,0011 --out out")
