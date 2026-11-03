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
    from envflag import env_on, env_raw, FALSE_WORDS   # 假值词表唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage1\pipeline.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on, env_raw, FALSE_WORDS

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1 import wires as wires_mod          # noqa: E402
from stage1 import pintrace as ray_mod         # noqa: E402
from stage1 import nets as nets_mod            # noqa: E402
from stage1 import pincol as pincol_mod        # noqa: E402
# ★ 「未连接脚」的 X 标记（`STAGE1_PIN_XMARK`，默认 0）：检出的 X 中心就是引脚外端。
from stage1 import xnc as xnc_mod               # noqa: E402
from stage1 import pstub as pstub_mod           # noqa: E402
# ★ 队友分支 `codex/improve-pin-network-parsing` 的引脚证据链（2026-10-16 并入，默认关）：
#   `pincolor` = 在图像上实测红色桩/绿线转折得到「彩色已确认的引脚端点」；
#   `pin_evidence` = 逐脚量六项图像证据（框覆盖率/墨迹连续长度/文字重叠/分支率…）。
#   它们产出的 `doc["pin_color"]` / `doc["pin_evidence"]` 正是 stage2 那五个
#   「引脚车道裁片 / 读数」读取器的**输入**（没有它，那些读取器实测是 no-op）。
from stage1 import pin_evidence as evidence_mod  # noqa: E402
from stage1 import pincolor as color_mod       # noqa: E402
from stage1.detect import Detector             # noqa: E402
from stage1 import detect as det_mod           # noqa: E402

DET = None   # 每个进程一份


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


# 大 IC 几何补检（`stage1/bigic.py`）。默认 **on**：YOLO 对 `box` 类的召回不够
# （220 张里 960 个 GT IC 框有 446 个整颗没检出），而一颗 IC 的键配不上就带走它
# 全部引脚 —— 引脚权重 0.25，一张图直接从 25 分掉到 0。实测代价 0.02 s/张。
BIGIC_ON = env_on("STAGE1_BIGIC", "1")

# ★★ sym16 小符号检测器（`stage1/sym16.py`）。**两条臂都默认关**，等 A/B 定档：
#   `STAGE1_SYM16`     = S1 新增提案（只追加「与既有件都不重复」的无名新件）
#   `STAGE1_SYM16_FIX` = S2 已有件精修（用同位置的 sym16 同型框换 `bbox_img`/`bbox_json`）
#   §5.532: `STAGE1_SYM16_FIX` 默认已翻 1 —— 13 页 82.2543 -> 82.3402 (+0.0859)，
#   池 28.5735 -> 28.6670 (+0.0934, 5 升/1 降/24 平)，comp 轴 3.9109 -> 3.8292。
#   只改已有件的 bbox（不动 key/Name/value、不增删件）；`STAGE1_SYM16` 仍默认 0
#   （13 页 -0.25 / 加件这条路历史上每次都是负的）。
# 为什么值得接：判分机 `metrics.py:351` 的坐标匹配只让 GT 的 `∅NNN` 件参与，
# 所以「多找的符号」只在 gnd/v/端口族上有收益（详见 `stage1/sym16.py` 顶注）；
# 而 `bbox_mismatch` 那部分分只能靠「改已有件的框」拿到。两个开关都不设时本模块
# 连 import 都不会发生 ⇒ 13 页产物逐位不变。
SYM16_ON = env_on("STAGE1_SYM16", "0")
SYM16_FIX_ON = env_on("STAGE1_SYM16_FIX", "1")

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
PIN_COLOR_ON = env_on("STAGE1_PIN_COLOR", "0")
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
# ★★ 黑白框实测引脚（队友 PR#3：`stage1/pin_evidence.py` 新增的
#   `recover_frame_pins`/`_frame_outer`/`_black_ink`）—— 把「按几何/文字猜出来的」
#   引脚点校准到**实测的框边笔画行**上。它只校准点、不增删引脚，且内部有三道
#   ValueError 守卫：不许改元件索引集合、不许改旧引脚索引、不许动已实测的彩色触点。
#   默认关：先 A/B（§5.114）；配 `STAGE1_MONO_FRAME_PINS=1` 打开。
MONO_FRAME_PINS_ON = env_on("STAGE1_MONO_FRAME_PINS", "0")
PIN_DIRECTION_GUARD = env_on("STAGE1_PIN_DIRECTION_GUARD", "0")
PIN_ATTACHMENT_GUARD = env_on("STAGE1_PIN_ATTACHMENT_GUARD", "0")
# ★★ 逐脚证据判决的**消费端**（两个门控、默认都 0 = 逐位历史行为）。
#   `stage1/pin_evidence.py` 只判决、不动引脚表（模块头一句就是
#   "it never moves or removes an output pin"），所以「筛掉它判不过的槽点」这件事
#   必须由 `prune_pin_evidence` 在这里做。**只筛不搬**：一个坐标都不改
#   （§5.508 实测把点重摆到腿长/关掉导线端点吸附/腿长唯一化，case11 掉 4.5~7.3 分）。
#   13 页离线实测（`work\_pev.py`，位置口径 pin F1）：
#     R0 现状 412 槽点 → 0.4478；只留 `trust=="trusted"` 306 槽点 → 0.5433
#     （`trust!="trusted"` 的 169 条里 5px 命中 **0** 条 = 全是外推/导线端点吸附）。
PINEV_TRUST_ON = env_on("STAGE1_PINEV_TRUST", "1")
#   在 A 之上再做「同侧同行去重」：同一 cid、同一 `edge`、切向落进同一
#   `pincol.MERGE_SNAP` 桶的多条槽点只留 `normal_ink_hits` 最大的一条
#   （相等留先出现者）。桶宽直接复用 `MERGE_SNAP`，不引入新常数。
PINEV_ROW_ON = env_on("STAGE1_PINEV_ROW", "0")
# ★★ 孪生槽点合并（`STAGE1_TWIN_MERGE`，默认 `"0"` = 关）：stage1 会给同一 `cid`
#   产出两个几乎重合的槽点（13 页 35 对，其中 3 对距离恰好 0.0；`work\_twins.py <tag> 4.0`）。
#   孪生槽让「号↔槽」失去 2D 分辨力（§5.540：全局指派被这件事拖回最近邻贪心）。
#   档位：
#     1 = 只合并**坐标完全相同**（距离恰好 0.0）的对 —— 不引入任何新常数、零风险；
#     2 = 合并同一 `cid` 内**任意两侧**、距离 < `pincol.MERGE_SNAP` 的对
#         （桶宽复用既有常数，`STAGE1_PINC_SNAP`，默认 5.0；不新增数值常数）。
#   保留 `(trust 等级, confidence, normal_ink_hits)` 字典序高者，同分留下标小者；
#   锁步删 `comp_pins[cid]` 与 `pin_evidence[cid]`，并由 `prune_pin_evidence` 重编号 nets。
#   ⚠️ 同族先例 `STAGE1_PINEV_ROW=1`（同 edge 切向 5px 桶）13 页 **−1.67** ⇒ 必须默认关、
#   且两档分开测（`work\_cv.py tw1/tw2 --env STAGE1_TWIN_MERGE=n`）。
_twin_raw = env_raw("STAGE1_TWIN_MERGE", "0").lower()
TWIN_MERGE = 0 if _twin_raw in FALSE_WORDS else (2 if _twin_raw == "2" else 1)
# ★★ 环境门控臂 A1/A2（两个都默认关 ⇒ 不设 env 时逐位等于现状）。
#   池（`200_train_cases` 30 案池）上 `std::type=box` 的 IC 件**一个引脚都不交付**，
#   根因是 `pintrace.py:42 INK = 170` 是**绝对**灰度阈值：池里 KiCad 风格渲染的图形层
#   （本体矩形灰 171、引脚腿 175/189）刚好被它挡在门外（纸底 243），只有文字能看见；
#   于是 `pin_evidence._frame_edge` 四侧 `cov=0.000` ⇒ `calibrated=False` ⇒ `_one_pin`
#   落到 `actual_frame_unconfirmed` ⇒ trust 恒 `uncertain` ⇒ 上面默认开的
#   `prune_pin_evidence`（`STAGE1_PINEV_TRUST=1`）把该件全部引脚删光。
#   A1：墨迹阈值改成**页面自适应** `max(pintrace.INK, Otsu)` —— 只放宽不收紧，
#   所以 13 页这类「Otsu < 170」的页面构造上与历史逐位相同。
INK_ADAPT_ON = env_on("STAGE1_INK_ADAPT", "0")
#   A2：在**证据层**做单调并集（严格掩膜 + 自适应掩膜各算一次逐脚证据，逐脚取
#   trust 更高的一条）。取高是单调算子 ⇒ 当前 trusted 的脚结构上不可能被降级。
#   A2 用的两个掩膜都从原图现算，**不依赖 A1**（两臂解耦）。
PINEV_UNION_ON = env_on("STAGE1_PINEV_UNION", "0")
# `build_pin_evidence` 的 trust 取值只有这三个（见 `pin_evidence._one_pin`）：
# 并集按这个偏序取高，同级再比 confidence。
_TRUST_RANK = {"trusted": 2, "uncertain": 1, "rejected": 0}


def _otsu_ink_threshold(gray):
    """`max(pintrace.INK, Otsu)`；退化图退回 `pintrace.INK`（常数，即历史行为）。

    `max` 保证**只放宽不收紧**：Otsu 低于常数的页面（13 页那批）阈值原样 170，
    产物与历史逐位相同。`0.5` 是**退化守卫不是调参**：墨迹占比过半说明这张图
    不是「纸底 + 墨」的两峰分布，Otsu 的谷已经失去意义，此时退回常数阈值。
    """
    base = int(ray_mod.INK)
    try:
        otsu = float(cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    except Exception:                                  # noqa: BLE001
        return base
    if not math.isfinite(otsu):
        return base
    threshold = max(base, int(otsu))
    if float((gray < threshold).mean()) > 0.5:         # 退化守卫：墨迹占比过半
        return base
    return threshold


def _ink_mask(gray, adapt=None):
    """灰度 → 二维墨迹掩膜；`adapt=None` 时取门控 `STAGE1_INK_ADAPT`。

    门控关闭时 `(gray < int(pintrace.INK))` 就是历史那一行，逐位不变。
    """
    enabled = INK_ADAPT_ON if adapt is None else bool(adapt)
    threshold = _otsu_ink_threshold(gray) if enabled else int(ray_mod.INK)
    return (gray < threshold).astype(np.uint8)


def _union_pin_evidence(strict_records, adapt_records):
    """逐脚取 trust 更高的一条记录（同级取 confidence 高）。

    **不增删条目、不改坐标**：返回的每个 cid 的列表长度、顺序与 `point_img` 都与
    输入一致（`prune_pin_evidence` 靠这个下标与 `comp_pins` 锁步删）。两个掩膜
    看见的是同一份 `comp_pins`，所以 `point_img` 必然相同；其余字段（`frame_coverage`
    / `normal_ink_hits` / `reason`）取自**胜出的那一条整条记录**，不跨记录拼接。
    """
    merged = {}
    for cid, records in (strict_records or {}).items():
        other = (adapt_records or {}).get(cid)
        if (not isinstance(records, list) or not isinstance(other, list)
                or len(other) != len(records)):
            merged[cid] = records          # 对不齐就原样退回：宁可不动，不许错位
            continue
        picked = []
        for strict_rec, adapt_rec in zip(records, other):
            if not isinstance(strict_rec, dict) or not isinstance(adapt_rec, dict):
                picked.append(strict_rec)
                continue
            strict_key = (_TRUST_RANK.get(str(strict_rec.get("trust")), -1),
                          float(strict_rec.get("confidence") or 0.0))
            adapt_key = (_TRUST_RANK.get(str(adapt_rec.get("trust")), -1),
                         float(adapt_rec.get("confidence") or 0.0))
            picked.append(adapt_rec if adapt_key > strict_key else strict_rec)
        merged[cid] = picked
    return merged


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


# ===== 以下两函数逐字取自队友 `codex/combined-pin-recognition`（PR#3）
#       `stage1/pipeline.py`；我们这边只把它们接在引脚定稿之后、build_nets 之前。


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


def prune_pin_evidence(pins, evidence, nets, trust_only=False, row_dedup=False,
                       twin_merge=0):
    """用逐脚证据判决**筛掉**引脚槽点，并把它在下游的三张表同步过滤。

    `pins` 与 `evidence` 逐位对齐（`build_pin_evidence` 是
    `[_one_pin(point) for point in pins[cid]]`，两个 overlay 也都按 index 写回），
    所以这里**只按下标同步过滤，绝不按坐标猜**。删脚会让下标整体前移，因此
    `nets[*]["pins"]` 里的 `[comp_id, pin_index]` 必须跟着**重编号**
    （`stage2/assemble.py` 拿它当 `num_of`/`pin_pk` 的键，不重编号就指到隔壁脚上）。
    删脚不改其余脚的键（脚号来自每脚读数，不是序号）。

    **只作用于 `evidence` 里有键的 cid**，其余元件逐位不变；长度对不齐的 cid
    一律跳过 = 退回历史行为（宁可不动，也不许把两张表错位）。

    `trust_only`：丢掉 `trust != "trusted"` 的槽点。
    `row_dedup`：同一 cid、同一 `edge`、切向落进同一 `pincol.MERGE_SNAP` 桶的
    槽点只留 `normal_ink_hits` 最大的一条（相等留先出现者）。
    `twin_merge`：同一 cid 内几乎重合的「孪生槽点」只留一条。`1` = 只合并**距离恰好
    0.0**（同一坐标，不可能合并掉两只真脚）；`2` = 合并**任意两侧**、距离
    < `pincol.MERGE_SNAP` 的对。保留 `(_TRUST_RANK[trust], confidence,
    normal_ink_hits)` 字典序最大者，同分留下标小的；坐标取 `pins` 的原值
    （`evidence.point_img` 只有 1 位小数）。

    返回 `(pins, evidence, nets, stat)`；`stat` 只用于打点（`trust`/`row`/`twin`
    条数、`comps` 件）。
    """
    if not (trust_only or row_dedup or twin_merge):
        return pins, evidence, nets, {}
    snap = float(getattr(pincol_mod, "MERGE_SNAP", 0.0) or 0.0)
    twin_level = 0 if not twin_merge else (1 if int(twin_merge) <= 1 else 2)
    keep_at = {}                       # cid -> {旧下标: 新下标}（只收真被筛过的件）
    stat = {"trust": 0, "row": 0, "twin": 0, "comps": 0}
    for cid, records in (evidence or {}).items():
        point_list = (pins or {}).get(cid)
        if not isinstance(point_list, list) or not isinstance(records, list):
            continue
        if len(records) != len(point_list):
            continue
        keep = [True] * len(records)
        if trust_only:
            for i, rec in enumerate(records):
                if isinstance(rec, dict) and rec.get("trust") != "trusted":
                    keep[i] = False
                    stat["trust"] += 1
        if row_dedup and snap > 0.0:
            winner = {}                # (edge, 桶号) -> (normal_ink_hits, 先出现的下标)
            for i, rec in enumerate(records):
                point = rec.get("point_img") or [] if isinstance(rec, dict) else []
                if not keep[i] or len(point) != 2:
                    continue
                edge = rec.get("edge")
                tangent = point[1] if edge in ("left", "right") else point[0]
                key = (edge, int(round(float(tangent) / snap)))
                hits = int(rec.get("normal_ink_hits") or 0)
                if key not in winner or hits > winner[key][0]:
                    winner[key] = (hits, i)
            best = {i for _hits, i in winner.values()}
            for i in range(len(records)):
                if keep[i] and i not in best:
                    keep[i] = False
                    stat["row"] += 1
        if twin_level and (twin_level == 1 or snap > 0.0):
            # 档 1 的判据是「恰好 0.0」，不是「小于某个小常数」⇒ 不新增阈值。
            limit = snap if twin_level == 2 else 0.0
            inclusive = twin_level == 1
            order = []                 # 按「该留下的优先」排序: (排名键, 下标, 坐标)
            for i, rec in enumerate(records):
                if not keep[i] or not isinstance(rec, dict):
                    continue
                point = point_list[i]
                if not (isinstance(point, (list, tuple)) and len(point) == 2):
                    point = rec.get("point_img") or []
                if len(point) != 2:
                    continue
                try:
                    point = (float(point[0]), float(point[1]))
                    key = (-_TRUST_RANK.get(str(rec.get("trust")), -1),
                           -float(rec.get("confidence") or 0.0),
                           -float(rec.get("normal_ink_hits") or 0.0), i)
                except (TypeError, ValueError):
                    continue
                order.append((key, i, point))
            order.sort()
            survivors = []
            for _key, i, point in order:
                twin = False
                for kept_point in survivors:
                    d = math.hypot(point[0] - kept_point[0], point[1] - kept_point[1])
                    if d < limit or (inclusive and d <= limit):
                        twin = True
                        break
                if twin:
                    keep[i] = False
                    stat["twin"] += 1
                else:
                    survivors.append(point)
        if all(keep):
            continue
        index_map, kept_points = {}, []
        for i, ok in enumerate(keep):
            if ok:
                index_map[i] = len(kept_points)
                kept_points.append(point_list[i])
        keep_at[cid] = index_map
        pins[cid] = kept_points
        evidence[cid] = [records[i] for i, ok in enumerate(keep) if ok]
        stat["comps"] += 1
    if not keep_at:
        return pins, evidence, nets, stat
    new_nets = []
    for net in nets or []:
        refs = net.get("pins") if isinstance(net, dict) else None
        if not refs:
            new_nets.append(net)       # 没有引脚引用的 net 一个字节都不动
            continue
        kept_refs = []
        for ref in refs:
            try:
                cid, old_index = ref[0], int(ref[1])
            except (TypeError, ValueError, IndexError):
                kept_refs.append(ref)
                continue
            index_map = keep_at.get(cid)
            if index_map is None:
                kept_refs.append([cid, old_index])
            elif old_index in index_map:
                kept_refs.append([cid, index_map[old_index]])
        # 即使一条引用都没被删，下标也可能整体前移 ⇒ 一律用重编号后的新列表
        shaped = dict(net)
        shaped["pins"] = kept_refs
        new_nets.append(shaped)
    return pins, evidence, new_nets, stat

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

    # ---- 空白框过滤（`STAGE1_BLANK_FILTER` = 墨迹下限，0=关）----
    # 用户观察（§5.283）：白底上会「扫出电容」。官方 10 的 255 个预测元件里有 37 个
    # 框内墨迹 <2%（8×20 的电阻框整块空白）——这些是纯 fp，删掉只赚不亏。
    _blank = float(os.environ.get("STAGE1_BLANK_FILTER", "0") or 0.0)
    if _blank > 0.0 and comps:
        _gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        _kept, _nb = [], 0
        for _c in comps:
            _b = _c.get("bbox_img") or []
            if len(_b) < 4:
                _kept.append(_c)
                continue
            _x0, _y0, _x1, _y1 = [int(round(float(t))) for t in _b[:4]]
            _crop = _gray[max(0, _y0):_y1, max(0, _x0):_x1]
            _ink = float((_crop < 200).mean()) if _crop.size else 1.0
            if _ink < _blank:
                _nb += 1
                continue
            _kept.append(_c)
        if _nb:
            print("[stage1] blank filter 丢 %d 个（墨迹<%.3f），留 %d"
                  % (_nb, _blank, len(_kept)), flush=True)
        comps = _kept

    # ---- 双尺度检测（`STAGE1_DUAL_DETECT` = 第二遍的 imgsz，0=关）----
    # 动机（§5.280）：整条管线换到 2560 时 `pin` tp 94→97、`netLine` 385→390（框更准），
    # 但 `component` tp 233→228（权重按 1600 训练，放大掉小目标召回），降 conf 只捞回 1 个。
    # 所以在**同一次**推理里跑两遍：1600 保召回，2560 精化框坐标；两遍按 IoU 配对，
    # 配对上的用 2560 的坐标，只在 2560 出现的补进来（保住召回缺口）。
    _dual = float(os.environ.get("STAGE1_DUAL_DETECT", "2560") or 0.0)
    _dual_fine_hi = float(os.environ.get("STAGE1_DUAL_FINE", "0.75") or 0.5)
    _dual_new_lo = float(os.environ.get("STAGE1_DUAL_NEW", "0.35") or 0.5)
    n_dual_fine = n_dual_new = 0
    _dual_maxpx = float(os.environ.get("STAGE1_DUAL_MAX_PIXELS", "0") or 0.0)
    _dual_minside = float(os.environ.get("STAGE1_DUAL_MIN_SIDE", "1400") or 0.0)
    _h0, _w0 = float(bgr.shape[0]), float(bgr.shape[1])
    if (_dual >= 320 and abs(_dual - float(imgsz)) > 1.0
            and (_dual_maxpx <= 0 or _h0 * _w0 <= _dual_maxpx)
            and max(_h0, _w0) >= _dual_minside):
        try:
            from stage1.detect import Detector as _Det2, to_json_box as _toj2  # noqa: F401
            _d2 = _Det2(weights, imgsz=int(_dual), conf=conf, device=device)
            _c2, _h2, _t2, _n2 = _d2.detect(bgr, conf=conf, use_cv=use_cv, tiler=tiler)
            _H2 = float(bgr.shape[0])

            def _iou(a, b):
                ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
                ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
                if ix1 <= ix0 or iy1 <= iy0:
                    return 0.0
                inter = (ix1 - ix0) * (iy1 - iy0)
                ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
                return inter / ua if ua > 0 else 0.0

            _taken = set()
            for _c in _c2:
                _bb = _c.get("bbox_img")
                if not _bb:
                    continue
                _best, _bj = 0.0, None
                for _j, _o in enumerate(comps):
                    if _j in _taken:
                        continue
                    _v = _iou(_bb, _o.get("bbox_img") or [0, 0, 0, 0])
                    if _v > _best:
                        _best, _bj = _v, _j
                if _bj is not None and _best >= _dual_fine_hi:
                    _taken.add(_bj)
                    _o = comps[_bj]
                    _o["bbox_img"] = [round(float(t), 2) for t in _bb]
                    if _c.get("bbox_raw_img"):
                        _o["bbox_raw_img"] = [round(float(t), 2) for t in _c["bbox_raw_img"]]
                    _o["bbox_json"] = [round(float(t), 2) for t in _toj2(_bb, _H2)]
                    if _c.get("center_img"):
                        _o["center_img"] = [round(float(t), 2) for t in _c["center_img"]]
                    _o["dual_fine"] = round(_best, 3)
                    n_dual_fine += 1
            # 只在第二遍出现、和第一遍配不上的框：补进来（保住 2560 掉的召回）
            for _c in _c2:
                _bb = _c.get("bbox_img")
                _ious = [(_j, _iou(_bb, comps[_j].get("bbox_img") or [0, 0, 0, 0]))
                         for _j in range(len(comps))] if _bb else []
                if os.environ.get("STAGE1_DUAL_DBG2"):
                    _mx = max(_ious, key=lambda t: t[1]) if _ious else (-1, 0.0)
                    print("[dbg dual2] %s %s iou_max=%.3f vs %s (n=%d)"
                          % (_c.get("id"), _c.get("type") or _c.get("name"), _mx[1],
                             (comps[_mx[0]].get("id"), comps[_mx[0]].get("type"))
                             if _mx[0] >= 0 else None, len(comps)), flush=True)
                if not _bb or any(v >= _dual_new_lo for _, v in _ious):
                    continue
                _new = dict(_c)
                _new["id"] = "d%04d" % n_dual_new
                _new["source"] = "yolo_dual"
                _new["bbox_img"] = [round(float(t), 2) for t in _bb]
                _new["bbox_json"] = [round(float(t), 2) for t in _toj2(_bb, _H2)]
                comps.append(_new)
                n_dual_new += 1
            if os.environ.get("STAGE1_DUAL_DETECT_DBG"):
                print("[dbg dual_detect] fine %d / new %d (imgsz %d)" % (n_dual_fine, n_dual_new, int(_dual)))
        except Exception as e:                                     # noqa: BLE001
            traceback.print_exc()
            errs.append(f"dual_detect: {type(e).__name__}: {e}")

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

    # ---- S1 `STAGE1_SYM16`（默认关）：sym16 检出的**无名新件**追加（add-only）----
    # 落点与上面 bigic 完全相同（在 `extract_wires` / `pintrace.trace_from_boxes` 之前，
    # 新件才会被涂掉、才会自己长出引脚射线）。只追加，绝不改任何既有件。
    sym16_stat = {}
    _s16_dets = None
    if SYM16_ON or SYM16_FIX_ON:
        # 两臂共用一次推理：S2 在 `symbox.grow` 之后才跑，但检测结果与那一步无关，
        # 所以这里算一次、把 `dets` 传下去（否则同一页要跑两遍 1280 的权重）。
        try:
            from stage1 import sym16 as sym16_mod
            _s16_dets = sym16_mod.raw_dets(bgr, conf)
        except Exception as e:                                     # noqa: BLE001
            traceback.print_exc()
            errs.append(f"sym16.det: {type(e).__name__}: {e}")
    if SYM16_ON and _s16_dets is not None:
        try:
            sym16_stat = sym16_mod.add_components(bgr, comps, hyps, conf=conf, dets=_s16_dets)
        except Exception as e:                                     # noqa: BLE001
            traceback.print_exc()
            errs.append(f"sym16: {type(e).__name__}: {e}")
            sym16_stat = {}

    # ★ `STAGE1_SYMBOX_EARLY`（§5.503，默认 `port`）：把 `symbox.grow` 提到**引脚/网络装配之前**。
    #   历史上它在文档构建前才跑，于是 `pins` 是按**小框**摆的；`grow` 把符号框撑到整颗之后，
    #   13 页 131 对端口符号里 **63% 的脚落进了框内**（GT 是 **0%**，中位在框外 5.3px）。
    #   提前跑 = 引脚从一开始就按长过的框摆（也顺带改了 `occlude` 的框，会动导线骨架）。
    #   档位证据（13 页，基线 `chk501` 81.05 / 79.82 / judge 77.43）：
    #     `port`（默认）= 只提前 `PORT_TYPES`（net_input/net_output/net_short/net_bidirection）
    #        ⇒ 81.17 / 79.94（**+0.12**）/ judge 77.57（**+0.14**），PASS，逐页仅 case1 +0.71、
    #        case12 +1.14，其余 11 页 +0.00；池 30 案 Δ +0.06（扣 0091 抖动后 ≈0）。
    #     `1`（全部符号）= 80.07（+0.25）/ judge 77.71（+0.28），但 case12 −0.75、case2 −0.43
    #        ⇒ 收益来自 `gnd`/`v` 也提前长框且**导线侧**的框一起变（WARN）。
    #     `1 + PREOCC + PRERAY`（导线侧冻结在长框之前）= 81.05 / 79.82（+0.01）⇒ 那部分收益全在导线侧。
    _sym_early = False
    _pre_bb = {}
    from . import symbox as _symbox                                  # noqa: PLC0415
    _sym_mode = str(os.environ.get("STAGE1_SYMBOX_EARLY", "port") or "port")
    if _sym_mode not in ("0", "", "off"):
        _only_port = _sym_mode.lower().startswith("port")
        # `STAGE1_EARLY_PREOCC` / `_PRERAY`：把长框之前的 bbox 存一份，供导线遮挡/射线继续用
        # （= 只让引脚看到长过的框，导线侧行为逐位不变）。`all` 档当年在 case12 掉 0.75 分
        # 就是导线侧被改了框，这条开关用来把两件事拆开。
        if (str(os.environ.get("STAGE1_EARLY_PREOCC", "0")) not in ("0", "")
                or str(os.environ.get("STAGE1_EARLY_PRERAY", "0")) not in ("0", "")):
            _pre_bb = {str(c.get("id")): list(c.get("bbox_img"))
                       for c in comps if c.get("bbox_img")}
        try:
            _sub = ([c for c in comps
                     if str(c.get("type") or "").strip().lower() in _symbox.PORT_TYPES]
                    if _only_port else comps)
            _sb0 = _symbox.grow(bgr, _sub)
            _sym_early = not _only_port
        except Exception as _e:                                      # noqa: BLE001
            errs.append(f"symbox_early: {type(_e).__name__}: {_e}")

    # 文字**定位**属第一阶段（用户 m03802 第三条指令），内容识别才属第二阶段。
    # 两个用途：①涂掉再跑骨架 —— 文字是导线假阳性的最大来源；②给导线端点当锚点，
    # 没有它「悬空导线」判不出来（jlc 风格靠 net label 连网，只有元件框一个锚点时
    # 全部真导线都会被误判成悬空：实测悬空元件 75 → 1、suspect 占比 20.5% → 3.9%）。
    t_txt = 0.0
    texts2 = []
    if do_text:
        from cvparts.textbox import detect_boxes as _tb
        t1 = time.time()
        texts = _tb(bgr)
        t_txt = time.time() - t1
        # §5.208.4(a)：**解耦**用。`texts`（本函数下文所有用途：涂墨、导线锚点）保持默认阈值；
        # 这里额外按 `EDA_TEXTBOX_INK_WIDE` 再检一份，只存进 doc 供**读数**侧追加
        # （`stage2.text.read_stage1_texts`），不改掩膜、不改锚点。默认 0 = 关，行为逐位不变。
        try:
            import os as _os
            _wide = int(_os.environ.get("EDA_TEXTBOX_INK_WIDE", "0") or 0)
        except Exception:
            _wide = 0
        if _wide:
            from cvparts import textbox as _txb
            _old = _txb.INK
            _txb.INK = _wide
            try:
                texts2 = _tb(bgr)
            finally:
                _txb.INK = _old
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
        if _pre_bb and str(os.environ.get("STAGE1_EARLY_PREOCC", "0")) not in ("0", ""):
            _occ_boxes = [_pre_bb.get(str(c.get("id"))) or c["bbox_img"] for c in comps]
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
            # 墨迹掩膜：门控 `STAGE1_INK_ADAPT` 关着时就是历史那一行 `gray < INK`。
            ink = _ink_mask(gray)
            cboxes = [c["bbox_img"] for c in comps]
            if _pre_bb and str(os.environ.get("STAGE1_EARLY_PRERAY", "0")) not in ("0", ""):
                cboxes = [_pre_bb.get(str(c.get("id"))) or c["bbox_img"] for c in comps]
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
    color_overlays, pin_connected, pin_color, pin_mono = {}, {}, {}, {}
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
            _ink = _ink_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            # ★ 符号色掩膜（`STAGE1_PIN_LEG_COLOR`）：引线是暗红、导线是绿色，颜色能
            #   把两者分开，用来逐元件量引线长度（见 `stage1/pins.py::_color_leg_box`）。
            _sym = None
            if nets_mod.pins_mod.PIN_LEG_COLOR:
                _hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                _h, _s, _v = _hsv[:, :, 0], _hsv[:, :, 1], _hsv[:, :, 2]
                _sym = (((_h <= 10) | (_h >= 170)) & (_s >= 60) & (_v >= 50)).astype(np.uint8)
            comp_pins = nets_mod.build_pins(
                comps, w.get("segments") or [], texts=texts, ink=_ink, sym_ink=_sym)
            # ★★ 引脚**列枚举**：把「框外那一排引脚号 / 框内那一排引脚名」的
            # 等距格点补进 `comp_pins`。规则是**不丢弃**：枚举点附近已有导线触点
            # 就用触点（网络装配靠它），没有触点就用格点坐标（= 补出来的脚），
            # 剩下没被认领的触点原样保留。详见 `stage1\pincol.py` 的模块注释。
            if PINC_ON:
                try:
                    pincol_cols = pincol_mod.enumerate_pins(
                        bgr, comps, texts, ink=_ink, sym=_sym,
                        segs=w.get("segments"))
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
            # ★★ 黑白框实测引脚校准（队友 PR#3 `stage1/pin_evidence.py:399
            #   recover_frame_pins`，默认关 `STAGE1_MONO_FRAME_PINS=0`）：
            #   把「按几何/文字猜出来的」引脚点**校准到实测的框边笔画行**上，
            #   并把那条实测证据行留进 `pin_mono` 审计账本（下游 `overlay_mono_evidence`
            #   直接复用，不再二次搜索）。守卫：不许改元件索引集合、不许改旧引脚索引、
            #   不许移动已实测的彩色触点 —— 违反即抛 ValueError，被这里吞掉并留痕。
            if MONO_FRAME_PINS_ON:
                try:
                    comp_pins, pin_synth, color_overlays, pin_connected, pin_mono = \
                        apply_mono_frame_pins(comps, comp_pins, bgr, _ink, texts,
                                              pin_synth, color_overlays, pin_connected)
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"pin_mono: {type(e).__name__}: {e}")
            # ★★ 「未连接脚」的 X 标记（`STAGE1_PIN_XMARK`，默认 0；见 `stage1/xnc.py`）：
            #   这套渲染把符号引线画成符号色、导线画成网络色，而 NC 脚在引线外端画一个
            #   网络色的 X。官方 10 张实测：检出的 X 中心落在 GT 脚点 2.0px 内，而我们把
            #   那两根脚发在了 9.0px / 169.8px 外 ⇒ 这是**可测**的端点证据（腿长则不可测）。
            #   必须在 `build_nets` 之前，因为网络装配吃的是最终引脚序列。
            if xnc_mod.XMARK_ON:
                try:
                    comp_pins, xnc_stat = xnc_mod.snap_x_marks(comps, comp_pins, bgr)
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"pin_xmark: {type(e).__name__}: {e}")
            # ★★ 「引线桩」闸门（`STAGE1_PIN_STUB`，默认 1；见 `stage1/pstub.py`）：
            #   真引线桩从元件面一直外伸 ≥10px，而脚号数字贴面印、NC 的 X 压在腿端
            #   —— 这两样都不是桩。面外 6px 与 9px 处都没有墨迹的脚点直接丢（官方 10 张
            #   只丢 6 根，pin 0.8570→0.8595、总分 90.09→90.15；§5.397.6 的重复行一半来自此）。
            #   同样必须在 `build_nets` 之前（网络装配吃最终引脚序列）。
            # ★★ 「引线桩」闸门（`STAGE1_PIN_XSTUB`，默认 1；见 `stage1/pstub.py`）：
            #   真引线桩从元件面一直外伸 ≥10px，而脚号数字贴面印、NC 的 X 压在腿端
            #   —— 这两样都不是桩。面外 6px 与 9px 处都没有墨迹、且同侧 14px 内有邻居的脚点
            #   直接丢（官方 10 张只丢 4 根：case5 `U1 pin_7`、case8 `U1 pin_1/pin_12/pin_2`；
            #   总分 90.09→90.51，池 50 案 34.47→34.16；§5.397.8）。
            #   必须在 `build_nets` 之前，因为网络装配吃的是最终引脚序列。
            #   注意：键名不能叫 `STAGE1_PIN_STUB` —— 那个键已被 `stage1/pins.py:185`
            #   的几何补桩模式（`gapfill`）占用，撞名会连带关掉补桩（实测 90.09→87.70）。
            if pstub_mod.STUB_ON:
                try:
                    comp_pins, pstub_stat = pstub_mod.filter_stubs(comps, comp_pins, _ink)
                except Exception as e:
                    traceback.print_exc()
                    errs.append(f"pin_xstub: {type(e).__name__}: {e}")
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
                pin_synth=pin_synth,
                # ★★ 「抹字断口」定向桥接：`texts` 就是上面 `extract_wires(...,
                # erase=texts)` 抹掉的那批文本框（`STAGE1_TEXT_BRIDGE>0` 才生效）。
                # 只认「两个线端各贴住同一个被擦除文本框」的缺口，绝不认元件框 ——
                # 抬全局容差会把电容两块极板并成一坨（见 `stage1\labels.py` 注释）。
                bridges=([(t.get("bbox_img") if isinstance(t, dict) else t)
                          for t in (texts or [])
                          if (t.get("bbox_img") if isinstance(t, dict) else t)] or None),
                # ★ 放宽半径时的方向判据（`STAGE1_JOIN_NRM`）需要每个元件的外框，
                #   用来算「引脚的外法向」。默认不传也能跑（该判据默认关）。
                boxes={c.get("id"): c.get("bbox_img") for c in (comps or [])
                       if c.get("id") and c.get("bbox_img")})
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
            if PINEV_UNION_ON:
                # 两个掩膜都**从原图现算**，不借用 `_ink`：严格掩膜恒为
                # `pintrace.INK`，自适应掩膜按 Otsu 现算 ⇒ A2 与 A1 完全解耦。
                _ev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pin_evidence = _union_pin_evidence(
                    evidence_mod.build_pin_evidence(
                        comps, comp_pins, _ink_mask(_ev_gray, adapt=False), texts),
                    evidence_mod.build_pin_evidence(
                        comps, comp_pins, _ink_mask(_ev_gray, adapt=True), texts))
            else:
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
        # ★★ 证据判决**反哺引脚表**（两个门控臂，默认全关 ⇒ 这一段整体不执行，
        #   产物与历史逐位相同）。放在三个 overlay **之后**：证据与 `comp_pins`
        #   的逐位对齐到这一行才定稿。只筛不搬 —— 坐标一个都不动。
        if PINEV_TRUST_ON or PINEV_ROW_ON or TWIN_MERGE:
            comp_pins, pin_evidence, nets, _pinev = prune_pin_evidence(
                comp_pins, pin_evidence, nets,
                trust_only=PINEV_TRUST_ON, row_dedup=PINEV_ROW_ON,
                twin_merge=TWIN_MERGE)
            print("[pinev] trust=%d row=%d twin=%d comps=%d pins=%d"
                  % (_pinev["trust"], _pinev["row"], _pinev["twin"], _pinev["comps"],
                     sum(len(v) for v in comp_pins.values())))

    tm = dict(tm)
    tm["wires_s"] = round(t_w, 2)
    tm["rays_s"] = round(t_r, 2)
    tm["text_s"] = round(t_txt, 2)
    tm["bigic_s"] = round(t_big, 2)
    tm["pins_s"] = pin_s
    tm["total_s"] = round(time.time() - t0, 2)
    # ★ 电源/地/端口符号的框长到「整颗」（`stage1/symbox.py`，`STAGE1_SYM_BOX` 默认 1）：
    #   GT 的符号框覆盖整个符号（横杠+竖杆+箭头），我们给的是内腔/标签附近，逐坐标差
    #   20~33px ⇒ 元件通道直接判错（case1 配上对的 47 组里 15 组死在框）。
    #   用符号色连通块在窄窗口（14px）内补齐：13 张 131 个符号的合格率 89% → 98%。
    #   落在 stage1 是因为 `fusion/run.py::_assemble` 会用原始 s1_doc 重新 build_doc。
    if _sym_early:
        _sb = {"on": True, "n": 0, "grew": 0, "early": True}
    else:
        try:
            from . import symbox as _symbox
            _sb = _symbox.grow(bgr, comps)
        except Exception as _e:                                      # noqa: BLE001
            _sb = {"on": False, "error": f"{type(_e).__name__}: {_e}"}
    # ---- S2 `STAGE1_SYM16_FIX`（默认关）：已有件的框精修 ----
    # 落点在 `symbox.grow` **之后**：`pins`（`comp_pins`）与导线骨架都已定稿，本步只改
    # **交付框** ⇒ 影响面锁在元件通道，不动引脚/网络。合并规则见 `stage1/sym16.py`。
    sym16_fix = {}
    if SYM16_FIX_ON:
        try:
            sym16_fix = sym16_mod.refine_components(bgr, comps, conf=conf, dets=_s16_dets)
        except Exception as _e:                                      # noqa: BLE001
            traceback.print_exc()
            errs.append(f"sym16_fix: {type(_e).__name__}: {_e}")
            sym16_fix = {}
    if os.environ.get("STAGE1_SYM_BOX_DBG"):
        print("[symbox doc] %s" % [(c.get("id"), c.get("type"), c.get("bbox_img"))
                                   for c in comps if str(c.get("type") or "") in _symbox.PORT_TYPES])
    # ★ 端口符号的脚位同样死在「`symbox.grow` 之前」（§5.503）：`pins` 是按**小框**的法向
    #   `PIN_OUT_BOX` 摆的，`grow` 把框撑到整颗之后，脚就落进了框内（实测 13 页 131 对端口
    #   符号：我们的脚 **63% 在框内**，GT **0% 在框内**、中位在框外 5.3px）⇒ 导线端点与端口
    #   脚对不上，端口被挂到错的网/挂不上。这里把单脚端口符号的脚重新贴到**长过之后**的框边
    #   外 `STAGE1_PORT_PIN_OUT` px（只改位置，不改成员关系）。`0` = 关（默认）。
    if int(os.environ.get("STAGE1_PORT_PIN_FIX", "0") or 0) and comp_pins:
        _pout = float(os.environ.get("STAGE1_PORT_PIN_OUT", "6") or 6.0)
        _nfix = 0
        for _c in comps:
            _ct = str(_c.get("type") or "").strip().lower()
            if _ct not in _symbox.PORT_TYPES:
                continue
            _pl = comp_pins.get(str(_c.get("id")))
            _bb = _c.get("bbox_img")
            if not _pl or len(_pl) != 1 or not _bb or len(_bb) != 4:
                continue
            _px, _py = float(_pl[0][0]), float(_pl[0][1])
            _cx, _cy = (_bb[0] + _bb[2]) / 2.0, (_bb[1] + _bb[3]) / 2.0
            _hw, _hh = max(1.0, (_bb[2] - _bb[0]) / 2.0), max(1.0, (_bb[3] - _bb[1]) / 2.0)
            if abs(_px - _cx) / _hw >= abs(_py - _cy) / _hh:
                _nx = (_bb[0] - _pout) if _px < _cx else (_bb[2] + _pout)
                _ny = min(max(_py, _bb[1]), _bb[3])
            else:
                _ny = (_bb[1] - _pout) if _py < _cy else (_bb[3] + _pout)
                _nx = min(max(_px, _bb[0]), _bb[2])
            if abs(_nx - _px) + abs(_ny - _py) < 0.5:
                continue
            comp_pins[str(_c.get("id"))] = [[round(_nx, 1), round(_ny, 1)]]
            _nfix += 1
        if os.environ.get("STAGE1_PORTFIX_DBG"):
            print("[dbg portfix] 重贴 %d 个端口的脚（外扩 %.1fpx）" % (_nfix, _pout))
    # ★ 双尺度补检的重复件（§5.502）：`yolo_dual` 的框在**补检当时**与主检框比 IoU 只有
    #   0.29（补检循环比较的是 `symbox.grow` 之前的小框），`symbox` 把主检的符号框长到整颗
    #   之后两个框就几乎重合了（case12 两对 gnd，最终 IoU 0.52）⇒ 同一个符号在交付里发两颗。
    #   所以去重必须放在 `symbox.grow` **之后**：同类型 + 至少一颗来自 `yolo_dual` +
    #   IoU ≥ `STAGE1_DUP_IOU` 判同，留分数高的那颗（分数接近时优先留主检）。
    #   `0` = 关（默认，逐位等于历史行为）。
    #   `0.45` 已落默认（§5.502）：13 页上只命中 case12 的两对 gnd（各 +1.68 页分 /
    #   official +0.13 / 判题机 +0.09），池上无可复现影响（`STAGE1_DUAL_MIN_SIDE=1400`
    #   让补检在池的小图上根本不跑）。`0` = 关。
    _dup_iou = float(os.environ.get("STAGE1_DUP_IOU", "0.45") or 0.0)
    if _dup_iou > 0 and len(comps) > 1:
        def _dup_ov(a, b):
            ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
            ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
            if ix1 <= ix0 or iy1 <= iy0:
                return 0.0
            inter = (ix1 - ix0) * (iy1 - iy0)
            ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            return inter / ua if ua > 0 else 0.0

        _dead, _ndup = set(), 0
        for _i in range(len(comps)):
            if _i in _dead:
                continue
            for _j in range(_i + 1, len(comps)):
                if _j in _dead:
                    continue
                _a, _b = comps[_i], comps[_j]
                if str(_a.get("type")) != str(_b.get("type")):
                    continue
                if "yolo_dual" not in (str(_a.get("source")), str(_b.get("source"))):
                    continue
                _ba, _bc = _a.get("bbox_img"), _b.get("bbox_img")
                if not (_ba and _bc) or _dup_ov(_ba, _bc) < _dup_iou:
                    continue
                _sa = float(_a.get("score") or 0) - (0.02 if _a.get("source") == "yolo_dual" else 0)
                _sc = float(_b.get("score") or 0) - (0.02 if _b.get("source") == "yolo_dual" else 0)
                # 默认「留主检、丢补检」：补检的用途是补召回（`_dual_new_lo` 之上多出来的框），
                # 坐标精化已经由 `_dual_fine_hi` 那一遍做过；两颗重合时主检的框更贴近本体。
                if os.environ.get("STAGE1_DUP_MODE", "main") != "score":
                    if (_a.get("source") == "yolo_dual") != (_b.get("source") == "yolo_dual"):
                        _loser = _j if _a.get("source") != "yolo_dual" else _i
                    else:
                        _loser = _j if _sa >= _sc else _i
                else:
                    _loser = _j if _sa >= _sc else _i
                _dead.add(_loser)
                _ndup += 1
                if _loser == _i:
                    break
        if _dead:
            for _k in sorted(_dead):
                comp_pins.pop(str(comps[_k].get("id")), None)
            comps[:] = [c for _k, c in enumerate(comps) if _k not in _dead]
        if os.environ.get("STAGE1_DUP_DBG"):
            print("[dbg dup] 去重 %d 颗（阈值 %.2f，剩 %d 件）" % (_ndup, _dup_iou, len(comps)))
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
        # ★★ 队友 PR#3 的黑白框实测账本（`STAGE1_MONO_FRAME_PINS=1` 时非空）：
        #   每条记录含 `added` / `calibrated` / `held` / `marker_rows` / `stroke_rows`
        #   —— 下游 `stage2/mono_pin_crops.py` 用 `stroke_rows` 直接取实测笔画行。
        #   关掉时恒为 `{}`（历史产物逐位不变）。
        "pin_mono": pin_mono,
        "nets": nets,
        # ★ 引脚接线的「留痕」：`orphans` = 够不着导线而被截断的引脚（历史上直接消失）、
        #   `weak` = 软判据区间内勉强挂上的引脚（带 `d`，下游可降权）。**不参与评分**，
        #   只是让「什么东西被丢掉了」在交付物里可见 —— 这正是 owner 要的「刷新」凭据：
        #   只有先看得见「哪些引脚没锁上」，才谈得上围绕它再扫一次。
        "pin_join": {"n_orphans": len(net_report.get("orphans", [])),
                     "n_weak": len(net_report.get("weak", [])),
                     "orphans": net_report.get("orphans", []),
                     "weak": net_report.get("weak", [])},
        # ★ 独家归属的账（业主 m25181：相邻两个框争一个位置，最终只归一个；
        #   「在其他任何地方，类似的情况也是同理」）。`rule` 是判据（默认
        #   `cd<=12` —— 19 张实测并 2 对、误伤 0），`dropped` 的主张**没有消失**，
        #   它以 `OWN_DEMOTE` 折进了 `kept` 那一槽的候选里（削弱，不覆盖）。
        "ownership": {"rule": det_mod.OWN_RULE, "demote": det_mod.OWN_DEMOTE,
                      "n_dropped": len(own_ledger), "ledger": own_ledger},
        "texts": [{"id": f"t{i:04d}", "bbox_img": [round(float(v), 1) for v in b]}
                  for i, b in enumerate(texts)],
        # §5.208.4(a) 解耦通道：`EDA_TEXTBOX_INK_WIDE` 打开的额外文字框（**只给读数用**，
        # 不参与本阶段的涂墨/锚点）。默认空列表。
        "texts_wide": [{"id": f"w{i:04d}", "bbox_img": [round(float(v), 1) for v in b]}
                       for i, b in enumerate(texts2)],
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
                  "n_sym16": int(sym16_stat.get("n") or 0),
                  "sym16": sym16_stat,
                  "sym16_fix": sym16_fix,
                  "n_errors": len(errs),
                  "auto_resolve": cnt},
    }
    return doc


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


# ---------------------------------------------------------------- 批
def _mark(r) -> str:
    """控制台标记：`ok` / `ok*N`（跑通但有 N 处被降级的异常）/ `FAIL`。"""
    if not r.get("ok"):
        return "FAIL"
    n = len(r.get("errors") or [])
    return f"ok*{n}" if n else "ok"


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


def _report_one(r):
    print(f"  {r['case']:<20}{_mark(r)}  "
          f"{r.get('timing', {}).get('total_s', '?')} s", flush=True)
    _blame(r)


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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("用 stage1/__main__.py 跑：python -m stage1 --cases 0050,0011 --out out")
