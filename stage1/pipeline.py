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
PIN_COLOR_ON = env_on("STAGE1_PIN_COLOR", "1")

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
        if confirmed == {"top", "right", "bottom", "left"}:
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
    """Keep evidence exactly aligned with the post-color pin sequence."""
    result = dict(evidence)
    for cid, recs in overlays.items():
        old = result.get(cid) or [{} for _ in recs]
        if len(old) != len(recs):
            old = [{} for _ in recs]
        result[cid] = [dict(r) if r is not None else dict(prior)
                       for prior, r in zip(old, recs)]
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
    bgr = cv2.imread(str(case.png_path), cv2.IMREAD_COLOR)
    if bgr is None:
        # Some Windows OpenCV builds cannot open Unicode dataset paths.
        try:
            encoded = np.fromfile(str(Path(case.png_path)), dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):
            bgr = None
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
        w, t_w = wires_mod.extract_wires(
            bgr, max_pixels=max_pixels,
            occlude=[c["bbox_img"] for c in comps], erase=texts, erase_pad=1.0,
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
    pin_evidence = {}
    color_overlays, pin_connected, pin_color = {}, {}, {}
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
            if PIN_COLOR_ON:
                try:
                    color_records = color_mod.extract_color_pins(bgr, comps)
                    comp_pins, pin_synth, color_overlays, pin_connected, pin_color = apply_color_pins(
                        comps, comp_pins, pin_synth, color_records)
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"pin_color: {type(e).__name__}: {e}")
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
                guard_attachments=PIN_ATTACHMENT_GUARD,
                pin_connected=pin_connected)
        except Exception as e:
            traceback.print_exc()
            # ★ 审计项②：这一条是**最贵**的吞异常 —— 它把 pin(0.25)+net(0.35)
            #   +line(0.10) = 0.70 权重整块清成 0，而这一层以前只 print 一下就继续。
            errs.append(f"pins/nets: {type(e).__name__}: {e}")
            comp_pins, nets, seg_groups = {}, [], []
        pin_s = round(time.time() - t_p, 2)

        # Evidence ranks the FINAL merged sequence. Failure must not erase
        # otherwise valid pins or nets; coordinates were finalized above.
        try:
            pin_evidence = evidence_mod.build_pin_evidence(
                comps, comp_pins, _ink, texts)
        except Exception as e:
            traceback.print_exc()
            errs.append(f"pin_evidence: {type(e).__name__}: {e}")
            pin_evidence = {}
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
        "pin_evidence": pin_evidence,
        "pin_color": pin_color,
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
        "timing": tm,
        # ★ 审计项②：非空就说明这张图的某一块被降级跑过（`stats` 里的 0 可能是
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
    """cases: list[Case]。workers>1 走进程池。"""
    import dataclasses
    jobs = [(dict(case_id=c.case_id, split=c.split,
                  png_path=c.png_path, json_path=c.json_path), str(out_dir), kw)
            for c in cases]
    results = []
    if workers <= 1:
        for j in jobs:
            r = _worker(j)
            results.append(r)
            print(f"  {r['case']:<20}{_mark(r)}  "
                  f"{r.get('timing', {}).get('total_s', '?')} s", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, j): j[0]["case_id"] for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"  {r['case']:<20}{_mark(r)}  "
                  f"{r.get('timing', {}).get('total_s', '?')} s", flush=True)
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("用 stage1/__main__.py 跑：python -m stage1 --cases 0050,0011 --out out")
