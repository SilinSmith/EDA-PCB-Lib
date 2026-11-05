"""第一阶段：元件几何。YOLO 通用检测 + 贴规范图章 + CV 专类部件，产出**多假设槽位**。

流程
----
1. **YOLO 推理**（`runs/detect/pcb43_v2_from_best/weights/best.pt`，conf 可调）。
   **imgsz 的类默认值是 1024，但正式管线用 1600**：`submission/run.py` 的
   `-imgsz` 默认就是 1600，`usage` 里记录的 220 张全量成绩全部是 1600 下跑的。
   1024 只是 `Detector()` 直接实例化时的兜底值（小图够用，大图上小元件会漏）。
2. **贴图章**（`schematic_parser/stamp.py`）：官方 GT 对符号类是**每类固定图章**
   （`r` 20 / `v` 10 / `c` 20 / `gnd` 20 / `testpoint` 20 …），而官方判据只要
   角点差 ≤20px ⇒ 把预测框换成「以预测中心为中心的规范图章」比让它自己回归尺寸更稳。
   逐实例测量的类（`box`/`switch`/`other`/`buzzer`/`block`/…）保持原框。
3. **CV 专类部件**（`cvparts/*`）：目前只有 `testpoint`（YOLO 召回 0.011，
   CV 召回 0.721）。CV 结果与 YOLO 结果按中心距离去重，都进槽位。
4. **多假设**（`cvparts/hypo.py`）：每个元件产出一个 `<id>.type` 槽位；若同一位置
   有多个不同来源/不同子类型的候选，**全部保留**，交给后面的规则 / LLM 裁决。
   `<id>.key` 槽位本阶段**留空** —— 用户 m03069：「给名字要联合后面的文字识别与绑定」。

坐标：产出里 `bbox_img` 是图像像素（左上原点），`bbox_json` 是赛题坐标（左下原点）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
try:                                    # 作为包导入（python -m stage1 / submission\run.py）
    from envflag import env_on, env_str      # 「假值词表」唯一实现，见 eda-pcb/envflag.py
except ImportError:                     # 直接跑脚本（python eda-pcb\stage1\detect.py 自检）：
    import sys as _sys, os as _os       # `sys.path[0]` 是本目录，得把 eda-pcb\ 补上
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from envflag import env_on, env_str

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schematic_parser import stamp as stamp_mod          # noqa: E402
from cvparts import testpoint as cv_testpoint            # noqa: E402
from cvparts import own as own_mod                       # noqa: E402
from cvparts.hypo import CaseHypotheses                  # noqa: E402
from assets import yolo_weights as _yolo_weights         # noqa: E402

# **不再写死绝对路径**（原来指向 `Y:\workspace\EDA\...`，进容器必炸）。
# `assets.yolo_weights()` 按「环境变量 `YOLO_WEIGHTS` -> 包内相对路径」找；
# 这里用 `must_exist=False` 是为了让 `python -c "import stage1.detect"` 这类
# 只想要函数、不要权重的场景不被 FileNotFoundError 挡住 —— 真正构造
# `Detector()` 时仍会加载权重并在缺失时报错。
DEFAULT_WEIGHTS = _yolo_weights(must_exist=False)
# 同一个符号上「YOLO 的框」与「CV 的框」中心距离小于这个值就算同一件
DEDUP_PX = 12.0
# ★ 独家归属（业主 2026-10-06 m25181：「相邻的两个框……最终只会归属于某一个」）。
#   原来 `DEDUP_PX=12` 只用在**跨来源**（YOLO vs CV）判同上，同类同源的重复框只靠
#   IoU-NMS —— 而贴章之后同类框都是同一个正方形，IoU 对小图章是坏尺子（见
#   `cvparts/own.py` 的模块文档）。这里把「归谁」统一交给 `own`，**并且留账**。
#   实测（`work\tools\diag\_owneval.py`，19 张 stage1 产物、3975 对同类框对，
#   用「最近同类 GT 是不是同一颗」当真相）：
#       cd<=12   并 2 对，真重复 2 / 误伤真符号 0
#       cd<=20   并 46 对，真重复 25 / 误伤 21（官方 bbox 容差，**不能**拿来当并框半径）
#   ★★ 但 **19 张端到端实测把它否掉了**（`work\tools\eval\fast.py --no-vlm`，
#   臂 A `STAGE1_OWN=off` 23.17 / 臂 B `cd12` 23.13，唯一变化是 `0103` **−0.77**）。
#   逐框追查（`work\tools\diag\_owndiff.py`）：并掉的是 `0103` 的两对 **gnd**，
#   中心差 9.41px / 9.11px —— 它们是**两颗各自独立的 GND 符号**（挨得很近但都配上
#   了不同的 GT，并掉后 tp 313→311）。`_owneval.py` 之所以报「误伤 0」，
#   是因为它的真相代理「最近同类 GT 是不是同一颗」在**密集排布**时会退化 ——
#   两个预测的**最近** GT 会塌到同一颗上，于是把「两颗真符号」误判成「一颗重复」。
#   ⇒ **代理不可信，端到端才算数。** 所以默认回到 `off`：机制留着、账留着、
#   `frac`/`cd12` 两种判据都留着，等有更可靠的判据（比如「同一颗 GT 的
#   Hungarian 归属 + 与 GT 的匹配唯一性」）再开。
OWN_RULE = env_str("STAGE1_OWN", "off").lower()
# 判负者的主张折进赢家槽位时打的折扣（只削弱，不覆盖）
OWN_DEMOTE = float(os.environ.get("STAGE1_OWN_DEMOTE", "0.85"))
CV_PARTS = {"testpoint": cv_testpoint}
# ultralytics 的 `model.predict` 自带 NMS，阈值写死 iou=0.7 —— 同一符号上两个框的
# IoU 正好卡在 0.70 附近时会**双双活下来**（0003 的 c0006/c0018 同类 IoU=0.7025，
# 就是这一条；它同时造成「一个 GND 被算成两个」的多检和正样本被挤掉）。这里统一
# 用我们自己的阈值再筛一遍，**tiler 与整图两个分支都要过**。
NMS_IOU = float(os.environ.get("STAGE1_NMS_IOU", "0.5"))

# ★ 类型级丢弃：整类预测几乎全是 fp 的，直接把该类元件从 stage1 拿掉（连同它的引脚、
# 它在网络里的引用）。`work\tools\diag\_compextra.py` 在 220 张上的逐类账
# （fp / tp / fn，以及「该类预测里 fp 的占比」）：
#     pin          172 /    2 /  173   ← 98.9% 是 fp
#     testpoint     68 /    1 /   42   ← 98.6% 是 fp
#     mosfet_npn    12 /    0 /    6   ← 100%
#     led          138 /   41 /  144   （77.1%，丢了反而略亏）
#     switch       115 /   50 /  107   （69.7%，同上）
# 只丢前三类的 220 张端到端实测（`work\_filt_C_plus_npn`）：
#     component F1 0.5288 → **0.5339**（tp 5320→5318、fp 4414→4161）
#     总分 25.0117 → **25.1647（+0.153）**；pin / netH / net_line 一个数没动。
# 复跑验证（`fast.py`）：10GT(9) 47.67 **逐位不变**（这些类在那 10 张里不出现），
# 19 张 28.31 → **28.35**（component 0.5226→0.5239），220 张 25.0117 → **25.1647**。
# 关掉：`STAGE1_DROP_TYPES=off`（或 `none`/`0`/空串）—— `env_str` 会把所有假值词归一成空串。
_STAGE1_DROP_ENV = env_str("STAGE1_DROP_TYPES", "pin,testpoint,mosfet_npn")
DROP_TYPES = frozenset(t.strip().lower() for t in _STAGE1_DROP_ENV.split(",") if t.strip())

# ★ CV 专类候选在**同槽位**竞争中直接胜出（默认 off）。
#   为什么需要：槽位竞争只会在「所有候选折叠后同类」时自动决议（`cvparts/hypo.py`
#   的 `auto_resolve`）；跨类（YOLO 说 `pin`、CV 说 `testpoint:hole`）一定不决议，
#   于是下面 `d["name"]` 回落分支把 CV 的主张整体丢掉。实测（`work\_tpA_base`，
#   `STAGE1_DROP_TYPES=none`）：`0011` 有 13 个槽位 CV 已报 `testpoint:hole`
#   （score 0.428~0.473、`rule:unanimous`）却没有对应元件；`0050` 的
#   `c0032.type` / `c0035.type` 是 `testpoint:ring3` 0.850 输给 `pin` 0.615。
#   打开：`STAGE1_CV_WIN=1`。**默认 off ⇒ 现有 19 张定点档结果逐位不变。**
_CV_WIN_ENV = env_str("STAGE1_CV_WIN", "off").lower()
CV_WIN = _CV_WIN_ENV not in ("", "0", "off", "false", "none", "no")

# ★ 逐类 conf 阈值（默认**关闭**：一个 `STAGE1_CONF_<CLASS>` 都不设时，行为与
#   以前全类统一 `conf=0.25` 逐字节相同）。
#   为什么需要：全类统一切 0.25 时 `gnd` 的低分正样本被一刀砍掉。13 页 open_GT 的
#   conf=0.02 原始检测（`work\_detdump_13.json`，审计脚本 `work\_detcls2.py`）逐类
#   GT 命中数（阈值 0.25/0.20/0.15/0.12/0.10/0.05）：
#       gnd  83 71 71 74 79 79 80   ← 0.12 比 0.25 多 9 颗，唯一有大空间
#       led  12  8  8  9 10 11 11   ← 0.10 多 1 颗
#       c 80/80  r 61/60  v 51/49  box 31  l 11/10  d 4 —— 0.25 附近已饱和
#       net_input 8→1、switch/pin/net_output 在 0.25 就是 0
#   误检（20px 内无任何 GT 件）：gnd 10 个（其中 conf>=0.25 的 3 个）、
#   net_input 9 个（>=0.25 的 6 个）、d 1、v 1 —— 所以只有 gnd 值得单独降。
#   把 4 个低分 gnd（0.152/0.146/0.126/0.077）补回预测，13 页 **+0.256**
#   （case1 +0.69、case11 +2.64）；这是**离线**账。⚠ 实测这笔账在真实管线上**不成立**：
#   13 页实跑 `STAGE1_CONF_GND=0.12` ⇒ 85.5091（base 85.6952，**−0.1861**），
#   `=0.10` ⇒ 85.4631（−0.2321）——多出来的低分 gnd 多半是「该颗已被检出」的重复件
#   （case4 离已有 gnd 14.8px、case5 13.9px），所以**默认必须关**，只有配合归并才有得谈
#   （见下面 `MERGE_PX` 块：加 12px 归并仍 −0.1012）。
#   ★ 实现要点（两个坑）：
#   1) 光在 `raw()` 之后过滤**不生效** —— 那 4 个框分数 <0.25，`model.predict`
#      根本不会吐出来。所以 `detect()` 里另算一个**推理下限**
#      `min(conf, 各逐类阈值)`，只降下限不动 `self.conf`（`self.conf` 仍是交付档
#      的 0.25，`Detector` 构造参数与 `pipeline.DET` 的复用判据都不受影响）。
#      低分框进 NMS 池**不会挪动**高分结果：`nms()` 按分数降序贪心，低分框只排在
#      高分框后面被裁，`>= conf` 那批的存活集合与不降下限时逐一相同。
#   2) 过滤放在 NMS **之后**、贴章**之前**：贴章后的第二次 NMS、CV 去重、`c%04d`
#      编号全部只看过滤后的列表 ⇒ 关掉时逐字节不变；开着时少了几个框，其后各类的
#      `c%04d` 编号整体前移（id 只是键名，评测按几何匹配，不吃亏）。
#   ⚠ 形态必须「一类一个 env」（`STAGE1_CONF_GND=0.12,STAGE1_CONF_LED=0.10`）：
#   `work\_onoff13.py` 的 env 串是**按逗号切分**的，`STAGE1_CONF_MAP="gnd=0.12,…"`
#   那种写法在那边会被切成两个非法 env，传不进来。
_CONF_PREFIX = "STAGE1_CONF_"
CONF_PER_CLASS: dict[str, float] = {}
for _k, _v in os.environ.items():                     # 键名小写化后与 `d["name"]` 对齐
    if not _k.upper().startswith(_CONF_PREFIX) or len(_k) <= len(_CONF_PREFIX):
        continue
    try:
        _t = float(_v)
    except (TypeError, ValueError):
        continue
    if 0.0 < _t < 1.0:                               # 非法值静默忽略，绝不改默认行为
        CONF_PER_CLASS[_k[len(_CONF_PREFIX):].lower()] = _t

# ★ 逐类中心距归并（默认**关闭**：`STAGE1_MERGE_<CLASS>` 一个都不设时不改任何行为）。
#   为什么需要：`gnd` 的重复检出是真实存在的。13 页 conf=0.02 原始检测
#   （`work\_detdump_13.json`）按「同类中心距 <12px 归并、留高分」重算：case5 33→10、
#   case1 36→20、case11 15→6、case13 4→1。
#   ⚠ 但它在 conf=0.25 档**零效果**（`merged12` 与 base 逐案同分、四轴全 0）：那些 1~9px
#   的重复在第一次 `nms(raw, NMS_IOU=0.5)` 里就被 IoU 抑制，进不到这里。真正要吃的重复是
#   **降阈值之后**新冒出来的低分件：case11 距已有 gnd 6.9px（归并后 case11 从 −0.61 回到
#   +0.49）、case4 14.8px、case5 13.9px（12px 半径吃不到，这两页 −1.07 / −0.98 纹丝不动）。
#   ⇒ 半径要 >14.8 才够用；硬上界来自 GT：13 页里相邻两颗真 gnd 的最小中心距 **20.00px**
#   （只有 case11 一处，其余页最小 96.8px），故 16 可用、20 是上限。
#   不归并就降逐类阈值的代价（13 页实跑，base 85.6952）：`STAGE1_CONF_GND=0.12` ⇒ 85.5091
#   （−0.1861）、`=0.10` ⇒ 85.4631（−0.2321）；加 12px 归并 ⇒ 85.5940（−0.1012）。
#   归并后的逐类曲线（13 页 gnd，20px 一对一匹配）：conf 0.25 ⇒ tp74/fp1；0.15 ⇒ 77/1；
#   0.12 ⇒ 80/1（precision 0.988 / recall 0.964）；0.10 ⇒ 80/2。
#   ★ 顺序：**先归并、再卡阈值** —— 先把同一颗的重复折成一个（留高分），再按类卡分。
#   ⚠ 阈值是**中心距**（欧氏，px），不是 IoU：同一颗被反复检出时平移只有 3px、框大小却
#   随贴章/朝向变，IoU 在这种情形上不给区分度。保留的是**高分那个的原始框**（不重算、不融合）。
#   写法同逐类阈值：`STAGE1_MERGE_GND=12`（设 0 表示该类显式关闭）。
_MERGE_PREFIX = "STAGE1_MERGE_"
MERGE_PX: dict[str, float] = {}
for _k, _v in os.environ.items():
    if not _k.upper().startswith(_MERGE_PREFIX) or len(_k) <= len(_MERGE_PREFIX):
        continue
    try:
        _t = float(_v)
    except (TypeError, ValueError):
        continue
    if _t > 0.0:
        MERGE_PX[_k[len(_MERGE_PREFIX):].lower()] = _t


def to_json_box(b, H: int):
    """图像框（左上原点）→ 赛题框（左下原点）。"""
    x1, y1, x2, y2 = [float(t) for t in b]
    return [x1, H - y2, x2, H - y1]


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def nms(dets, iou_thr: float = 0.6):
    """同类贪心 NMS，按 score 降序。dets: list[dict] 带 cls/box/score。"""
    keep = []
    for c in sorted({d["cls"] for d in dets}):
        grp = sorted([d for d in dets if d["cls"] == c], key=lambda d: -d["score"])
        while grp:
            best = grp.pop(0)
            keep.append(best)
            grp = [d for d in grp if _iou(best["box"], d["box"]) <= iou_thr]
    return keep


def merge_close(dets, px: dict):
    """同类中心距 < px[class] 的重复检出归并，只留 score 最高的那个。

    `px` 形如 `{"gnd": 12.0}`（见文件顶 `MERGE_PX`）；空 dict 时原样返回。
    返回列表**保持入参顺序**（只删不排）：`detect()` 里 `d["_i"]` / `c%04d`
    编号依赖这个顺序，重排会让"没被归并的类"也换 id。
    """
    if not px:
        return dets
    keep, seen = set(), {}
    for d in sorted(dets, key=lambda d: -d["score"]):
        k = str(d["name"]).lower()
        t = px.get(k)
        if t is None:
            keep.add(id(d))
            continue
        b = d["box"]
        c = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
        got = seen.setdefault(k, [])
        if any((c[0] - q[0]) ** 2 + (c[1] - q[1]) ** 2 < t * t for q in got):
            continue
        got.append(c)
        keep.add(id(d))
    return [d for d in dets if id(d) in keep]


class Detector:
    """加载一次 YOLO，反复用。多进程时每个进程各持一份。"""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, imgsz: int = 1024,
                 conf: float = 0.25, device: str = "0"):
        # ★ 先设显存配额再碰 ultralytics/torch —— LLM 在同卡上常驻占 ~10GB。
        try:
            from gpu_mem import cap_vram
            cap_vram()
        except Exception:                                           # noqa: BLE001
            pass
        from ultralytics import YOLO
        self.weights = str(weights)      # ★ 审计项④：缓存复用时要能比对配置
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.names = [str(n) for n in self.model.names.values()] \
            if isinstance(self.model.names, dict) else [str(n) for n in self.model.names]

    def raw(self, bgr: np.ndarray, conf: float | None = None):
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf if conf is None else conf,
                               device=self.device, verbose=False)[0]
        out = []
        if r.boxes is None or not len(r.boxes):
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sc = r.boxes.conf.cpu().numpy()
        for b, c, s in zip(xyxy, cls, sc):
            out.append({"box": [float(v) for v in b], "cls": int(c),
                        "name": self.names[int(c)] if int(c) < len(self.names) else str(int(c)),
                        "score": float(s), "source": "yolo"})
        return out

    # ------------------------------------------------------------------
    def detect(self, bgr: np.ndarray, conf: float | None = None,
               use_cv: bool = True, tiler=None, own_ledger: list | None = None):
        """返回 (components, hypotheses, timing, raw_count)。

        `tiler` 形如 ``(tile, overlap)`` 时切块推理并做跨切片全局 NMS
        （大图小符号的对策；实测收益有限，见 `docs/官方口径评测报告.md`）。

        `own_ledger` 传一个 list 进来，就会把**独家归属**的逐条判负记录写进去
        （谁、输给谁、差多少、按哪条规则），供 `stage1/pipeline.py` 落盘。
        """
        H, W = bgr.shape[:2]
        # ★ 逐类阈值（见文件顶 `CONF_PER_CLASS`）：没设任何 env 时 `_floor is _conf`、
        #   下面那段过滤整块跳过 ⇒ 与旧的 `conf=0.25` 档逐字节相同。
        _conf = self.conf if conf is None else conf
        _floor = min([_conf] + list(CONF_PER_CLASS.values())) if CONF_PER_CLASS else _conf
        t0 = time.time()
        if tiler:
            tile, ov = tiler
            raw = []
            step = max(64, tile - ov)
            for y in range(0, max(1, H), step):
                for x in range(0, max(1, W), step):
                    x2, y2 = min(W, x + tile), min(H, y + tile)
                    x1_, y1_ = x2 - tile, y2 - tile
                    x1_, y1_ = max(0, x1_), max(0, y1_)
                    sub = bgr[y1_:y2, x1_:x2]
                    if sub.size == 0:
                        continue
                    for d in self.raw(sub, _floor):
                        b = d["box"]
                        d["box"] = [b[0]+x1_, b[1]+y1_, b[2]+x1_, b[3]+y1_]
                        raw.append(d)
            raw = nms(raw, NMS_IOU)
        else:
            raw = nms(self.raw(bgr, _floor), NMS_IOU)
        t_yolo = time.time() - t0

        # ★ 逐类中心距归并 / 逐类 conf 后过滤（都放在两次 NMS 之间、贴章之前）：
        #   先归并（同颗重复折成一个、留最高分），再按类卡分。
        #   每类阈值 `CONF_PER_CLASS[name]`，无该类的项就回落到 `_conf`（=0.25）。
        if MERGE_PX:
            raw = merge_close(raw, MERGE_PX)
        if CONF_PER_CLASS:
            raw = [d for d in raw
                   if d["score"] >= CONF_PER_CLASS.get(str(d["name"]).lower(), _conf)]

        # 贴图章
        #   ★ 2026-10-07 改动：图章由**正方形**换成 **每类一个长方形 + 轴向**
        #     （`stamp.py` 顶注有实测：正方形平均角点误差 2.74px / 长方形 0.00px）。
        #     轴向从 `box_raw`（检测器自己回归出的框）的宽高比推断 —— 所以必须在
        #     贴章**之前**把 `box_raw` 存下来，并且**用 `box_raw` 去贴**，不要用
        #     当前那个可能已经被前一轮贴过的 `d["box"]`。
        for i, d in enumerate(raw):
            d["_i"] = i                       # 贴章后要重排，先用它记住原始顺序
            d["box_raw"] = list(d["box"])
            d["box"] = stamp_mod.snap_box(d["box_raw"], d["name"])
        # ★ 贴章之后**再压一次同类重复**（2026-07 审计 A2）。
        #   `snap_box` 把同类的框全都换成**同一个模板**（同尺寸、同轴向）、以原框
        #   中心为中心（见 `schematic_parser/stamp.py:snap_box`），于是两个原始框
        #   IoU < 0.5 因而都活下来的重复检测（同一个符号被检出两次、或本来框得一
        #   大一小），贴章之后会**完全重合**。原来只在贴章**之前**跑一次 NMS，
        #   这一批就漏成了 component 的 fp。贴章幂等、以中心为准 ⇒ 再压一次不会
        #   伤到真检测。
        #   ⚠ 长方形模板比正方形**窄**（`r` 宽 8 / 20、`l` 宽 6 / 34），所以「中心
        #   差 10px 的两个重复框」在正方形下还有点 IoU、在长方形下可能完全不重叠
        #   —— 这条 NMS 对窄模板的压制力弱于以前。原始框的 IoU 那一遍没有变。
        raw = nms(raw, NMS_IOU)
        raw.sort(key=lambda d: d.pop("_i"))   # 复原 original 顺序（`c%04d` 的编号要稳定）

        comps, hyps = [], CaseHypotheses(case="")
        # 1) YOLO 的候选
        for i, d in enumerate(raw):
            cid = f"c{i:04d}"
            d["id"] = cid
            cx = (d["box"][0] + d["box"][2]) / 2.0
            cy = (d["box"][1] + d["box"][3]) / 2.0
            hyps.add(f"{cid}.type", d["name"], d["score"], source="yolo",
                     evidence={"box_raw": [round(v, 1) for v in d["box_raw"]],
                               "snapped": [round(v, 1) for v in d["box"]]},
                     question=f"{cid}（中心 {cx:.0f},{cy:.0f}）是什么元件？",
                     context={"cx": round(cx, 1), "cy": round(cy, 1)})
        # 2) CV 专类候选：以 **子类型** 作为候选值（用户 m03069：「当成不同的几种
        #    类型……最后合并到同一个类去算」），折叠规则登记在 folds 里。与 YOLO 的
        #    提案放在**同一个槽位**上做竞争；同位置的 YOLO 框不限于同类 —— 跨类才是
        #    真正需要裁决的歧义（比如 YOLO 说 `v`、CV 说 `testpoint:hole`）。
        t_cv = 0.0
        n_cv_new = 0
        if use_cv:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
            for cname, mod in CV_PARTS.items():
                _fold_to = cname
                for h in mod.detect_hypotheses(gray):
                    t_cv += time.time() - t1
                    t1 = time.time()
                    label = f"{_fold_to}:{h['subtype']}"
                    hyps.fold(label, _fold_to)
                    best, best_d = None, 1e18
                    for d in raw:
                        bcx = (d["box"][0] + d["box"][2]) / 2.0
                        bcy = (d["box"][1] + d["box"][3]) / 2.0
                        dist = max(abs(bcx - h["cx"]), abs(bcy - h["cy"]))
                        if dist < best_d:
                            best, best_d = d, dist
                    if best is not None and best_d <= DEDUP_PX:
                        # 同一位置有 YOLO 的提案 -> 作为竞争候选并入该槽位
                        hyps.add(f"{best['id']}.type", label, h["score"],
                                 source=f"cv:{h['subtype']}", evidence=h["evidence"],
                                 context={"cv_subtype": h["subtype"],
                                          "dist_to_yolo": round(best_d, 2)})
                        # ★ `STAGE1_CV_WIN=1` 时让 CV 的专类标签直接锁定这个槽位
                        #   （否则跨类不决议，回落成 YOLO 的 `d["name"]`）。
                        if CV_WIN:
                            hyps.resolve(f"{best['id']}.type", label, by="cv:win")
                    else:
                        cid = f"v{n_cv_new:04d}"
                        n_cv_new += 1
                        side = stamp_mod.STAMP_SIDE.get(cname, cv_testpoint.STAMP)
                        box = [h["cx"] - side/2, h["cy"] - side/2,
                               h["cx"] + side/2, h["cy"] + side/2]
                        raw.append({"box": box, "box_raw": box, "cls": -1, "name": label,
                                    "score": h["score"], "source": f"cv:{h['subtype']}",
                                    "id": cid, "subtype": h["subtype"]})
                        hyps.add(f"{cid}.type", label, h["score"],
                                 source=f"cv:{h['subtype']}", evidence=h["evidence"],
                                 question=f"{cid} 是一个 {cname}（画法 {h['subtype']}）吗？",
                                 context={"cx": round(h["cx"], 1), "cy": round(h["cy"], 1)})
            t_cv = time.time() - t1 + t_cv

        # ★ 独家归属：一个位置只能有一个主人（业主 m25181）。
        #   位置在这里 —— YOLO 与 CV 的候选都进 `raw` 之后、建组件之前，
        #   所以它同时覆盖「同类同源重复」与「跨来源撞车」，是一条**统一**的规则
        #   （原来前者靠 IoU-NMS、后者靠 DEDUP_PX，两套尺子）。
        if OWN_RULE and OWN_RULE not in ("off", "none", "0"):
            parties = [own_mod.Party(id=d["id"], box=tuple(d["box"]), kind=str(d["name"]),
                                     score=float(d["score"]), meta={"det": d})
                       for d in raw]
            if OWN_RULE.startswith("frac"):
                kept = own_mod.thin(parties, ledger=own_ledger)
            else:
                kept = own_mod.legacy_thin(parties, ledger=own_ledger)
            # ★ 判负者的主张**不许消失**：把它那一槽的候选按 `OWN_DEMOTE` 折进
            #   赢家那一槽 —— 「可以削弱内容的置信度，但不能轻易覆盖」。
            for e in (own_ledger or []):
                los, win = e.get("dropped"), e.get("kept")
                sl = hyps.slots.get(f"{los}.type") if (los and win) else None
                if sl is None:
                    continue
                for a in list(sl.alts):
                    hyps.add(f"{win}.type", a.value, round(float(a.score) * OWN_DEMOTE, 4),
                             source=f"own:{a.source}", evidence=dict(a.evidence),
                             context={"lost_to": win, "dev": e.get("dev"),
                                      "rule": e.get("rule")})
            keep_dets = {id(p.meta["det"]) for p in kept}
            raw = [d for d in raw if id(d) in keep_dets]

        counts = hyps.auto_resolve()
        for d in raw:
            sid = f"{d['id']}.type"
            sl = hyps.slots.get(sid)
            _ctype = (hyps.fold_value(sl.chosen) if sl is not None and sl.resolved()
                      else hyps.fold_value(d["name"]))
            if DROP_TYPES and str(_ctype or "").strip().lower() in DROP_TYPES:
                continue                   # ★ 类型级丢弃，见文件顶 `DROP_TYPES`
            comps.append({
                "id": d["id"],
                "type": _ctype,
                "proposal": sl.chosen if sl is not None and sl.resolved() else d["name"],
                "score": round(float(d["score"]), 4),
                "source": d["source"],
                # CV 形状证据：CV 自建槽位是 `d["subtype"]`；YOLO 的框被 CV 复用
                # CV 形状证据：CV 自建槽位是 `d["subtype"]`；YOLO 的框被 CV 复用时是
                # 上面补的 `d["cv_subtype"]`。
                "subtype": (d.get("subtype") or d.get("cv_subtype")),
                "bbox_img": [round(float(v), 2) for v in d["box"]],
                "bbox_json": [round(float(v), 2) for v in to_json_box(d["box"], H)],
                # 贴图章**之前**的框。图章一律是正方形，所以「符号是横的还是竖的」
                # 这个信息在 bbox_img 里已经被抹掉了；而官方 GT 的引脚点位恰好是
                # 「bbox 中心 ±20 沿符号主轴」，需要这个长短边来定轴（见 stage2/README）。
                # YOLO 的训练标签来自官方 json 的**未贴章**框（r 8x20、c 16x20、l 34x6），
                # 所以它预测出来的 raw 框长短边是可信的。
                "bbox_raw_img": [round(float(v), 2) for v in d["box_raw"]],
                "center_img": [round((d["box"][0]+d["box"][2])/2, 2),
                               round((d["box"][1]+d["box"][3])/2, 2)],
                "key": None,
                "n_alts": len(sl.alts) if sl is not None else 1,
                "pending": bool(sl is not None and not sl.resolved()),
            })
        return comps, hyps, {"yolo_s": round(t_yolo, 2), "cv_s": round(t_cv, 3)}, counts


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import schematic_parser as S
    d = Detector(conf=0.25)
    ids = sys.argv[1:] or ["0050", "0011", "0102"]
    for cid in ids:
        c = S.find_case(cid, S.DEFAULT_DATA_ROOT)
        if c is None:
            print(f"{cid}: 找不到"); continue
        bgr = cv2.imread(str(c.png_path), cv2.IMREAD_COLOR)
        comps, hyps, tm, cnt = d.detect(bgr)
        from collections import Counter
        cc = Counter(x["type"] for x in comps)
        print(f"\n{cid}  {bgr.shape[1]}x{bgr.shape[0]}  元件 {len(comps)}  "
              f"YOLO {tm['yolo_s']}s  CV {tm['cv_s']}s  auto_resolve {cnt}")
        print("  类型:", dict(cc.most_common(12)))
        pend = [x for x in comps if x["pending"]]
        print(f"  待裁决槽位 {len(pend)}:")
        for x in pend[:4]:
            sl = hyps.slots[x["id"] + ".type"]
            print("   ", x["id"], [(a.value, round(a.score, 2), a.source) for a in sl.alts])
