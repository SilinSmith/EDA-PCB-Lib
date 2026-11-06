# -*- coding: utf-8 -*-
"""赛题提交驱动：`/submission/agent_system -input /input -output /output`。

官方契约（`work\\archive\\oci\\unpack\\...\\submission\\README.md`）
------------------------------------------------------------------
容器入口 `/submission/agent_system -input <dir> -output <dir>`，必须写出：

    /output/decision_log.json
    /output/final_results/<case_name>_Pred.json     # 每个输入图片一份
    /output/final_results/run.log                   # 可选
    /output/runs/                                   # 可选：中间产物

`<case_name>` = 图片文件名去掉扩展名，**可能带空格**（`0025-Altium Designer`）。

流程
----
    输入 PNG 目录
      -> 阶段一  stage1.pipeline.run_cases          -> runs/stage1/<case>.stage1.json
      -> 阶段二  stage2.run.process                 -> runs/stage2/<case>.json
                                                     + runs/stage2/_assigned/<case>.json
      -> 阶段三  stage3.run.process（LLM 仲裁，可选） -> runs/stage3/<case>.json
      -> VLM 补强（`-vlm`，默认 auto）               -> runs/stage3_vlm/<case>.json
      -> 融合层（`-fusion`，默认 on）                -> runs/fusion/<case>.json
      -> 导出    final_results/<case>_Pred.json + decision_log.json + run.log

导出取**最靠后存在的那一层**（融合 > VLM 补强 > 阶段三 > 阶段二），所以任何一层
整批失败都不会让 final_results 少产物。

**每张图独立 try/except**：一张图炸掉不能拖垮整批（评测是整目录跑，2 小时预算）。
★ 2026-10-14（用户 m42543「考虑各种灾难性现场」）—— 上面两条**曾经只是口号**，现已落实：
`main` = `parse_args → _fix_device → try _run except _safety_net`，`_safety_net` 会对输入目录
**每张图**补空壳 `_Pred.json` 并写 `decision_log.json`（`summary.fatal`）后返回 3；阶段一
`run_cases` 的进程池崩溃改在本进程串行补跑；导出层对没有产物的 case 写空壳（不是 `continue`）；
`_fix_device` 在无 CUDA 时把 `-device 0` 回退成 `cpu`；同名 case 去重；巨图走
`stage1/pipeline.py:_imread_bgr` 的降采样护栏。故障注入测试见 `tools/dev/_hardtest.py`。

★ 2026-10-15（用户 m42543「判题机同时给全部样例且允许并发」）—— **case 级扇出**：
`-case-workers`（默认 -1 = 自动：可见 GPU ≥2 时取 min(GPU 数, 图数)，否则 1 = 老行为）
把每张图的**全流程**分给一个独立子进程、每个子进程独占一张卡；父进程负责分派、
看门狗（`EDA_WALL_BUDGET_S`）、合并产物、给没产出的 case 补空壳并写 `decision_log.json`。
`-workers` 只并行阶段一（9 张实测只快 ~10%），而阶段二/三/融合层是逐 case 顺序循环，
交付口径 26.6 s/张（官方 case 1）~3.7 min/张（派生大图）⇒ 50 张必须并行才进 2 小时。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # `eda-pcb/`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import assets                                    # noqa: E402
from schematic_parser.data import Case           # noqa: E402
from stage2 import compbox                       # noqa: E402  `STAGE2_COMP_BOX`（占位框→字形模板，默认 median，见 compbox 头注释）

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _log(fp, msg: str):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def discover_inputs(input_dir: Path):
    """`<case_name>` 去扩展名。先看顶层；顶层没有图再递归找一层（官方是平的）。"""
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    if not files:
        files = sorted(p for p in input_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    return [(p.stem, p) for p in files]


def _dedupe_items(items):
    """按 `<case_name>` 去重（保序，保留排序后的第一个）。返回 `(kept, dropped)`。

    ★ 灾难现场加固（用户 m42543）：`a.png` + `a.jpg`（或递归到不同子目录的同名图）
    会让 `stage1` 输出**同名互相覆盖**（`-workers>1` 时还是并发写同一个文件，
    可能写出半个 JSON），最后只得到 1 份 `_Pred.json`，而 `run.py` 的导出计数
    与 `ok1 = {r["case"] ...}` 都察觉不到。同名输入在评测端本身就是歧义（
    `<case_name>_Pred.json` 只有一个名字），所以**保留第一个、把被丢弃的显式报出来**。
    """
    seen, kept, dropped = set(), [], []
    for cid, p in items:
        if cid in seen:
            dropped.append((cid, str(p)))
            continue
        seen.add(cid)
        kept.append((cid, p))
    return kept, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EDA 赛题六 提交入口")
    ap.add_argument("-input", "--input", dest="input_dir", required=True)
    ap.add_argument("-output", "--output", dest="output_dir", required=True)
    ap.add_argument("-limit", "--limit", dest="limit", type=int, default=0)
    ap.add_argument("-cases", "--cases", dest="cases", default="")
    ap.add_argument("-device", "--device", dest="device", default="0",
                    help="YOLO 用；torch 侧由 textrec.infer.normalize_device 翻译"
                         "（'0' -> 'cuda:0'，没卡则退 CPU）")
    ap.add_argument("-imgsz", "--imgsz", dest="imgsz", type=int, default=1600)
    ap.add_argument("-tile", "--tile", dest="tile", type=int, default=0,
                    help="阶段一切块推理的瓦片边长（px）。0（默认）= 整图，即今天的行为。"
                         ">0 时交给 `stage1.detect.Detector.detect(tiler=(tile, overlap))` "
                         "做跨瓦片全局 NMS —— 大图（如 val 的 4500×2850 / 3450×2280）"
                         "在 `-imgsz 1600` 下小元件与细脚文字会被缩没，切块是唯一"
                         "不看整图缩放的对策。**未过 val 前不要改默认**。")
    ap.add_argument("-overlap", "--overlap", dest="overlap", type=int, default=0,
                    help="切块重叠像素（仅 `-tile > 0` 时生效；0 = 自动取 tile/8）")
    ap.add_argument("-workers", "--workers", dest="workers", type=int, default=1,
                    help="**stage1 内部**进程池大小（CPU 密集；>1 会各自占显存）")
    ap.add_argument("-case-workers", "--case-workers", dest="case_workers", type=int,
                    default=-1,
                    help="**case 级扇出**：把整张图的全流程分给 N 个独立进程并行，"
                         "每个子进程独占一张卡（`-device` 递增）。-1（默认）= 自动："
                         "可见 GPU ≥2 时取 min(GPU 数, 图数)，否则 1（= 今天的行为）。"
                         "环境变量 `EDA_CASE_WORKERS` 可覆盖自动值；0 = 强制关。"
                         "用户 m42543：判题机一次给全部样例且允许并发；"
                         "交付口径每张 ~30 s（官方 case 1）~3.7 min（派生大图），"
                         "50 张串行顶穿 2 小时预算。")
    ap.add_argument("-no-wires", "--no-wires", dest="no_wires", action="store_true",
                    help="跳过骨架提线（最耗时的 CPU 段）")
    ap.add_argument("-texts", "--texts", dest="texts", default="both",
                    choices=("stage1", "detect", "union", "both", "auto"))
    ap.add_argument("-mode", "--mode", dest="mode", default="off",
                    choices=("auto", "off", "llm", "stub"),
                    help="阶段三：off（默认）= 什么都不做。llm 需显式指定 —— "
                         "实测 8 张 A/B 里 off 与 llm 逐项指标完全相同（0 改动），"
                         "而 llm 要 161.7 s/张（220 张 ~10 小时，顶穿 2 小时预算）。"
                         "详见 stage3/arbitrate.py 的 arbitrate() docstring。")
    ap.add_argument("-llm-batch", "--llm-batch", dest="llm_batch", type=int, default=8)
    ap.add_argument("-vlm", "--vlm", dest="vlm", default="off",
                    choices=("auto", "on", "off"),
                    help="阶段二之后的 VLM 补强（读 IC 位号 + 引脚号/名）。"
                         "**默认 off（2026-11-06，§5.592）**：本地端点把 VLM 全链打开"
                         "实测 13 页 84.83 / official 83.06，比离线档 87.8881 / 86.1632"
                         "低 3.06 / 3.10 分，还要吃 token（Score_Token 按总消耗排名赋分）；"
                         "auto = 先探一次端点收不收图再决定。")
    ap.add_argument("-vlm-workers", "--vlm-workers", dest="vlm_workers", type=int,
                    default=2, help="VLM 并发数。**默认 2**（2026-10-13，§5.229）："
                                    "40 张批上 1 并发与 2 并发**逐案逐位相同**（10GT 83.64），"
                                    "而 4 并发在同一个 40 张批上掉到 81.09（10GT comp f1 "
                                    "0.9955→0.9446），且客户端看不到任何失败/空答案/截断 —— "
                                    "是服务端在 4 路交错时把答案串了。2 并发比 4 只慢 ~25%"
                                    "（40 张批 515.5s vs 452.5s），换来读数可信。")
    ap.add_argument("-vlm-margin", "--vlm-margin", dest="vlm_margin", type=float,
                    default=60.0, help="读位号时元件框外扩的像素。")
    ap.add_argument("-vlm-value", "--vlm-value", dest="vlm_value",
                    default="on", choices=("on", "off"),
                    help="用 VLM 读分立元件（R/C/L/Y/F）的『值』覆盖管线结果。"
                         "**默认开**：关键是『位号一致』门控（只采纳模型自报位号与元件键"
                         "相符的那些读数）。220 张 A/B：不开 23.9559 → 开门控版 **24.9051**"
                         "（component tp 5232→5527，text_mismatch 961→666）；"
                         "不带门控的旧版是负的（23.6404）。详见 vlm/README.md。")
    ap.add_argument("-vlm-value-margin", "--vlm-value-margin", dest="vlm_value_margin",
                    type=float, default=80.0,
                    help="读『值』时元件框外扩的像素。别调小：45px 会把 100nF 的 F 切掉，"
                         "逐字命中率从 0.94 掉到 0.79。80 是实测峰值。")
    ap.add_argument("-vlm-value-batch", "--vlm-value-batch", dest="vlm_value_batch",
                    type=int, default=1,
                    help="**【请求合并】**把 N 格值裁片拼成一张图一次问完（1 = 关，旧行为）。"
                         "值通道原来 220 张要发 3492 次请求，是全管线第二大开销。"
                         "⚠️ 拼图会让每格变小，而值通道对裁片尺寸极其敏感"
                         "（margin 25 → 0.212 vs margin 80 → 0.9423）"
                         "⇒ **打开前必须先用 fast.py / _vlmab2.py 做 A/B**，"
                         "每格精度掉超过 1 个百分点就不要开。")
    ap.add_argument("-vlm-value-side", "--vlm-value-side", dest="vlm_value_side",
                    type=int, default=520, help="拼图时每格的边长（像素）。")
    ap.add_argument("-vlm-value-cols", "--vlm-value-cols", dest="vlm_value_cols",
                    type=int, default=2, help="拼图的列数。")
    ap.add_argument("-vlm-name", "--vlm-name", dest="vlm_name",
                    default="on", choices=("on", "off"),
                    help="用 VLM 读 IC 的『器件型号』写进 `Name`（官方 PDF 第 8 页：ic 比 Name）。"
                         "220 张 A/B：单独开这一项 23.9559 → **24.4969**（+0.541，component "
                         "text_mismatch 961→784，只花 1021 次调用）。")
    ap.add_argument("-vlm-name-margin", "--vlm-name-margin", dest="vlm_name_margin",
                    type=float, default=80.0,
                    help="读器件型号时元件框外扩的像素（40 起就平了）。")
    ap.add_argument("-vlm-pin", "--vlm-pin", dest="vlm_pin",
                    default="on", choices=("on", "off"),
                    help="用 VLM **引脚通道**（读 IC 的引脚号/名，替换引脚键）。"
                         "**默认开**（§5.404 起，主尺 = open_GT-1020）："
                         "该尺子上 9B `-vlm-pin on` **70.80** vs off **70.42**"
                         "（pin 0.5495→0.5650，13 张逐位可复现）；"
                         "只留「号」那一半（`-vlm-pin-name off`）只有 70.55。"
                         "§5.398 当年在**官方 10 张**上判负（9B 91.24 vs 89.13、pin 0.8854 vs 0.8198）——"
                         "那套图的引脚名本来就贴在框边、GT 脚名常等于号，换不到分；"
                         "open_GT 是干净的 datasheet 风格页，脚名是可读的真名 ⇒ 符号翻转。"
                         "想退回旧口径用 `-vlm-pin off`。"
                         "⚠ 这是**替换**引脚键那条路"
                         "（`VLM_PIN_MERGE=0`）；`VLM_PIN_MERGE=1` 是只并名的旧路线。")
    ap.add_argument("-vlm-pin-name", "--vlm-pin-name", dest="vlm_pin_name",
                    default="on", choices=("on", "off"),
                    help="引脚通道里**『名』那一半**（融合层 `FUSION_VLM_PIN_NAME`）。"
                         "关掉只剩「号」：官方 9 张 **+1.95 且九张零回归**（全开 +2.88）。"
                         "**如果评测集更像 JLC 派生的那 200 张（GT 引脚名常常就是号本身），"
                         "关掉这一半更稳。** 号那一半由 `-vlm-pin` 控制。")
    ap.add_argument("-vlm-desig-fallback", "--vlm-desig-fallback",
                    dest="vlm_desig_fallback", default="on", choices=("on", "off"),
                    help="对『键还是兜底键 pNNN』的元件补读位号（不限类型）。"
                         "220 张里这类元件 2601 个，其中 92%% 在 10px 内就有文字框，"
                         "但 `assign.py` 的全局去重把那个框判给了别人；而 pNNN 键"
                         "永远不可能和 GT 对上，换掉只有赚没有赔。")
    ap.add_argument("-vlm-desig-all", "--vlm-desig-all",
                    dest="vlm_desig_all", default="off", choices=("on", "off"),
                    help="位号补读的覆盖面：`off`（默认）= 只读 IC + 兜底键 `pNNN`；"
                         "`on` = **所有**带位号的元件都问一次 VLM（`enrich.py` 的 "
                         "`desig_all`）。动机：§5.219–§5.222 证明位号墙是唯一的瓶颈，"
                         "而 `desig_slots` 一个候选都没有的元件有 418 颗（56%）；"
                         "代价是位号调用数翻倍（20 张 461 次 → 约 1 000 次），"
                         "交付口径 24.08 s/张 会涨到约 40 s/张 ⇒ 220 张 88 分钟 → 约 150 分钟，"
                         "所以**默认关**，只在能证明赚钱时开。")

    ap.add_argument("-fusion", "--fusion", dest="fusion", default="on",
                    choices=("off", "on", "llm"),
                    help="融合层（21 节点架构的落地）：先验 + 解 + 合理性检查。"
                         "`on` = 确定性节点 **+ LLM 轮（judge/audit/rescan）**；"
                         "`llm` = 再叠加 cropfix（重裁重读）；"
                         "`off` = 不做融合，直接导出阶段三/二产物。"
                         "LLM 轮实测约 **+30 s/张**（case 1+10 墙钟：融合 3.5 s → "
                         "62.7 s；折算 220 张约 +1.8 h），临时关掉用 `FUSION_LLM=0`。"
                         "220 张实测：LLM 轮关 = 28.4973、开 = 28.4200、不融合"
                         "（VLM 补强后的阶段三）= 28.4133。默认带 LLM 轮是 owner "
                         "m25661 的决定（见下方 `fusion_rows` 上方那段注释）。")
    ap.add_argument("-llm-max-slots", "--llm-max-slots", dest="llm_max_slots",
                    type=int, default=0)
    ap.add_argument("-gpu-frac", "--gpu-frac", dest="gpu_frac", type=float,
                    default=0.0,
                    help="本进程可用显存占总量的比例（0 = 用环境变量 EDA_GPU_FRAC "
                         "的默认 0.30）。本地 LLM 常驻占 ~10GB/16GB，别调大。"
                         "设 EDA_GPU_CAP=0 可整体关掉。")
    ap.add_argument("-gpu-headroom", "--gpu-headroom", dest="gpu_headroom",
                    type=float, default=0.0,
                    help="改为「给 LLM 留出 N MiB」；优先级高于 -gpu-frac。")
    ap.add_argument("-lean", "--lean", dest="lean", action="store_true",
                    help="**精简档**（为赛题 `Score_Token` 排名项）：`-vlm-pin off` + "
                         "`-vlm-value-batch 4`。官方 10 张实测（`work\\_tokcost.py`，"
                         "protocol2）：默认 222 次 / 152 164 token / 118.2 s / 79.66 "
                         "→ 精简 **115 次 / 93 181 token（−39%）/ 94.9 s / 79.84**，"
                         "且逐项诊断与默认**同级**（component.text_mismatch 2 = 2）。"
                         "再加 `-vlm-value off` 更狠：**82 次 / 45 428 token（−70%）/"
                         "62.6 s / 79.45**（−0.21，落在跨运行 VLM 噪声带 ±0.38 内，"
                         "代价是 component.text_mismatch 2→5）。")
    ap.add_argument("-offline", "--offline", dest="offline", action="store_true",
                    help="**零外部调用档**（为赛题 `Score_Token` 排名项与「官方只给纯文本"
                         "API」两种现实准备）：`-vlm off` + `FUSION_LLM=0` + "
                         "`FUSION_ITERATE=0` + `STAGE1_WIRE_COLOUR=dominant` + "
                         "`STAGE2_PINROT=0`（**保留**融合层的确定性节点，不发请求，"
                         "值 +0.24），已由环境变量显式设过的值**不覆盖**。"
                         "官方 10 张实测：0 次调用 / 0 token / 22.6 s / protocol2 70.04"
                         "（默认档 222 次 / 152 164 token / 118.2 s / 79.66）。")
    a = ap.parse_args(argv)
    _fix_device(a)
    _apply_lean(a)
    _apply_offline(a)
    _auto_offline_if_no_endpoint(a)
    _apply_vlm_tuning(a)
    a.platform_budgets = _platform_budgets()
    print("[run] 护栏：%s" % " ".join("%s=%s" % kv for kv in a.platform_budgets.items()),
          flush=True)
    try:
        # ★ case 级扇出（用户 m42543）：`_run_fanout` 在扇出数 <= 1 时**原样**调
        #   `_run(a)`，所以单卡机器/小批量的行为与以前逐位一致。
        return _run_fanout(a)
    except Exception as e:                                          # noqa: BLE001
        return _safety_net(a, e)


def _apply_lean(a) -> None:
    """把 `-lean` 翻译成「省 token 但分量不掉」的通道组合。

    依据（`work\\_tokcost.py`，官方 10 张，protocol2，逐通道隔离 A/B）：
      · `-vlm-pin off`（关引脚/网格通道，省 23 次调用）：**+0.18 分**（79.84 vs 79.66），
        逐项诊断不劣（pin.missing 12 vs 13）⇒ 脚号/脚名本地（textrec + `BoxICLibrary`
        先验 + `STAGE2_PINROT`）已经读对，VLM 那一档是冗余。这与判题机报告
        `pin_number_accuracy_on_matched = 1.000`（§5.37）互证。
      · `-vlm-value-batch 4`（值通道 4 格拼一图，省 104 次调用）：**−0.36 分**
        （79.30 vs 79.66，落在 ±0.38 的跨运行噪声带内），`component.text_mismatch`
        与默认一致（2 = 2）⇒ 拼图没有伤到值文本。
    两者合并实测 **115 次 / 93 181 token / 94.9 s / 79.84**。
    """
    if not getattr(a, "lean", False):
        return
    a.vlm_pin = "off"
    try:
        if int(getattr(a, "vlm_value_batch", 1) or 1) < 4:
            a.vlm_value_batch = 4
    except (TypeError, ValueError):
        a.vlm_value_batch = 4
    print("[run] 精简档：-vlm-pin off｜-vlm-value-batch %s（官方 10 张：115 次 / "
          "93 181 token / 79.84，与默认档同级）" % a.vlm_value_batch, flush=True)


def _apply_vlm_tuning(a) -> None:
    """VLM 档才有意义的开关组合（离线档不动它）。

    ★ `STAGE2_DESIG_DIGITS=0`（「位号只读到数字」的补解，默认 1）：它会把 IC 旁边的
    **引脚号**（`1`/`2`）当成位号补成 `U1`/`U2`，从而抢占真正的 `U1`，让 VLM 后来读到的
    `U1`（挂在兜底键 `p001` 的 IC 上）被唯一性判据挡掉。`open_GT-1020` 13 张实测：
    关掉它 **69.51 → 70.42**（comp 0.8504→0.8590、pin 0.5324→0.5495）；而离线档里它是
    **正贡献**（68.28 → 67.97）⇒ 只在 VLM 档关。
    """
    if getattr(a, "offline", False) or str(getattr(a, "vlm", "")).lower() == "off":
        return
    if "STAGE2_DESIG_DIGITS" not in os.environ:
        os.environ["STAGE2_DESIG_DIGITS"] = "0"


def _apply_offline(a) -> None:
    """把 `-offline` 翻译成「真正一次外部调用都不发」的环境组合。

    为什么不能只靠 `-vlm off`：实测（`work\\_tokcost.py`，官方 10 张）`-vlm off` +
    `FUSION_LLM=0` + `STAGE2_PINROT=0` 仍然发出了 **19 次**调用、9 726 token ——
    两条旁路不归 `-vlm` 管：

    * `fusion/iterate.py:189`（⓪.9 定向回传：按 pincol 格点窗回问引脚名，`FUSION_ITERATE`）；
    * `stage1/wires.py:537`（导线色相提示，`STAGE1_WIRE_COLOUR=hint` 默认）。

    这 19 次值 **+1.11** 分（71.15 → 70.04 少的就是它），所以不能无条件砍掉 ——
    只有显式要「离线档」时才关。已在环境里设过的值不覆盖（便于单独做 A/B）。
    """
    # 默认档（`-vlm off`）也走这里：交付默认必须与实测口径一致 —— §5.592 实测 VLM
    # 全链 −3.06 / −3.10 分且额外吃 token，评测机上只要探到端点就会掉分。
    if not (getattr(a, "offline", False) or str(getattr(a, "vlm", "")).lower() == "off"):
        return
    a.vlm = "off"
    # ★ 只关 LLM 轮，**不关融合层**：融合层里的确定性节点（先验 + 解）本身不发请求，
    #   实测值 +0.24（`zero` 档 70.04 vs `-fusion off` 的 69.80，官方 10 张）。
    #   `-fusion on`（默认）不覆盖 `FUSION_LLM` ⇒ 这里显式置 0 即可。
    # ★★ `STAGE2_PINROT` **不再强制 0**（2026 修正）：它是**本地**识别器（引脚名 90/270°
    #   旋转重读，不发任何外部请求），历史那一版把它塞进离线档是为了对齐 §5.378 的口径，
    #   代价却是交付档比评测档低 **0.66**：官方 10 张 90.80 → **90.14**，掉分全在
    #   case6 94.01→91.78、case7 93.21→91.03、case10 90.31→**88.14**（pin 0.8696→0.7826）。
    #   实测 `STAGE1_WIRE_COLOUR=dominant` 本身零影响（90.80 逐轴相同），留着只为彻底
    #   绕开 `stage1/wires.py::_hint_wire_hue` 那条 VLM 旁路。
    # ★★ 2026 修正（§5.483）：离线档改为 `confirm`。`dominant` 会把**符号本身的颜色**
    #   当成导线色 —— case12 的暗红符号占主色相 h=35，整页导线于是画在符号笔画上
    #   （stage1 段 60% 红、交付导线只覆盖 GT 5.8%，该页 net_line = 0.000）。`confirm`
    #   先在元件框外做「长直线最多的色相」本地估计（`pskel.local_wire_hue`，无外部调用），
    #   再要求它在调色板里被证实，且**只在主色相是红系时**才改写 dominant（`pskel.py:534`）
    #   ⇒ 其余 12 页逐位不变。13 页主尺 +0.20（80.16 → 80.36，case12 57.08 → 59.63），
    #   池 30 案 +0.06（落在批次噪声 ±0.1 内，逐案无一致性伤害）。
    for k, v in (("FUSION_LLM", "0"), ("FUSION_ITERATE", "0"),
                 ("STAGE1_WIRE_COLOUR", "confirm")):
        if k not in os.environ:
            os.environ[k] = v
    print("[run] 离线档：-vlm off｜FUSION_LLM=%s｜FUSION_ITERATE=%s｜STAGE1_WIRE_COLOUR=%s"
          "｜STAGE2_PINROT=%s（本地，未被离线档改动；零外部调用，保留融合层确定性节点）"
          % (os.environ.get("FUSION_LLM"), os.environ.get("FUSION_ITERATE"),
             os.environ.get("STAGE1_WIRE_COLOUR"),
             os.environ.get("STAGE2_PINROT", "(默认 1)")),
          flush=True)


def _auto_offline_if_no_endpoint(a) -> None:
    """`-vlm auto`（默认档）开跑前先探一次端点；探不通就直接退到离线档。

    为什么必须这么做（2026 实测）：端点**存在但鉴权失败（HTTP 401）**时，`-vlm auto`
    会让所有 VLM 调用带着重试跑完整条链路，产物是**不确定的**——官方 10 张（默认档、
    无 `-offline`）实测 **89.20**，而同一张 case1 两次跑分别给出 88.93 / 87.08
    （唯一差别落在 `C154`↔`C14` 与一个电容值上），比离线档（90.80、逐字节可复现）低
    1.6 分且每次不同。判分机上「端点死/没有 key/模型不收图」都是大概率事件
    （`vlm/client.py::preflight` 的 docstring 记了两种已栽的现场），所以默认档必须先探。
    """
    if getattr(a, "offline", False):
        return
    if str(getattr(a, "vlm", "") or "").strip().lower() != "auto":
        return
    ok = False
    try:
        from vlm import client as _vc
        ok = bool(_vc.probe(timeout=8.0))
    except Exception as e:                                          # noqa: BLE001
        print("[run] ⚠️ 端点探测抛异常（%s: %s）⇒ 按探不通处理" % (type(e).__name__, e), flush=True)
    if ok:
        print("[run] 端点探测通过 ⇒ `-vlm auto` 保持开启", flush=True)
        return
    print("[run] ⚠️ 端点探不通（无端点 / 无 key / 不收图）⇒ **自动切离线档**："
          "`-vlm off` + FUSION_LLM=0 + FUSION_ITERATE=0，产物确定可复现"
          "（官方 10 张 90.80；带失败调用跑是 89.20 且逐次不同）", flush=True)
    a.offline = True
    _apply_offline(a)


def _fix_device(a) -> None:
    """**CPU 回退**（灾难现场加固，用户 m42543）。

    提交默认 `-device 0`；官方基线镜像里未必有 GPU。`ultralytics` 在 CUDA 不可用时对
    显式指定的 `device=0` 是**直接抛 ValueError**（`torch_utils.py:326-346`），而
    `stage1` 的 `_worker` 会把它吞成 `ok: False` ⇒ **220 张全失败、退出码 0、0 产物**。
    这条我们自己的 `textrec.infer.normalize_device` 早就做了，语义不一致，这里对齐。
    只改「看起来是卡号/cuda 的那种值」，显式 `cpu` 不动。
    """
    d = str(getattr(a, "device", "") or "").strip().lower()
    if not (d.isdigit() or d.startswith("cuda")):
        return
    try:
        import torch
        ok = bool(torch.cuda.is_available())
    except Exception:                                               # noqa: BLE001
        ok = False
    if not ok:
        print("[run] ⚠️ 请求 device=%r 但本机 torch.cuda.is_available()=False ⇒ "
              "回退 CPU（否则 YOLO 会对每张图抛 ValueError，整批 0 产物）"
              % (a.device,), flush=True)
        a.device = "cpu"


# ======================================================================
# 交付护栏：外部推理（VLM / LLM）的时间预算（用户 m40034「2 小时跑 50 样例」）
# ======================================================================
# 赛题硬约束（`docs/COMPETITION_RULES.md`）：**全部案例评测总时长 < 2 小时**，而且
# **未完成的案例双罚**（分子少它、`m/M` 再折一次）⇒ 「超时被杀」比「产出差」贵得多。
# 实测：交付口径 26.6 s/张（官方图）~ 3.7 min/张（派生大图），50 张串行顶穿预算 ⇒
# 平时靠 `-case-workers` 扇出（每个 lane 一个独立进程、独占一张卡）摊薄。
# 但扇出**摊不掉**「外部端点被限流 / 挂死」这种病：`vlm/client.ask()` 在没有护栏时
# **单次耗时没有上界** —— 黑洞端点（连得上、不回包）要等满 `VLM_TIMEOUT` 才抛错返回，
# 而「答得出来但正文空 / 被 length 截断」那条路会把 4 次重试串起来（旧默认 600 s ⇒
# 最坏 **2400 s = 40 min** 一次调用）；一个 lane 几张图就能把 2 小时整段吃掉，
# 而官方判题机是按墙钟硬杀的 —— 那时候没落进 `final_results/` 的案例全是双罚。
# 所以这里补三个**只在用户没设时**才生效的默认值，量级取「正常跑根本不会触发」：
#   · `VLM_TIMEOUT=180`：本机实测单次调用 0.2~9 s（VLM 段整张图 16.3 s）⇒ 180 s 宽裕；
#   · `VLM_BUDGET_S=1800`：**本进程**累计外部推理时间上限（`vlm/client.py` 实现：
#     到点后 `ask()` 直接返回空串降级成纯 CV，且单次超时会被剩余预算夹住）；
#   · `EDA_WALL_BUDGET_S=6900`（115 min）：父进程看门狗，到点强杀剩余 lane 并对没产出
#     的图**补空壳** —— 空壳只丢准确度分，不会触发 `m/M` 双罚。
# 想复现「没有护栏」的旧行为：`VLM_TIMEOUT=600 VLM_BUDGET_S=0 EDA_WALL_BUDGET_S=0`。
_PLATFORM_DEFAULTS = (
    ("VLM_TIMEOUT", "180"),
    ("VLM_BUDGET_S", "1800"),
    ("EDA_WALL_BUDGET_S", "6900"),
)

# 扇出时**全进程同时在飞**的 VLM 请求数上限（`_child_concurrency` 用它反推每 lane 的
# `-vlm-workers`）。8 是 §5.61 量出来的拐点：8 路在飞时官方 10 张逐字节不变，16 路时
# 答案本身变了（-2.79 分）。串行路径（lane 数 ≤ 1）不受它影响。
VLM_INFLIGHT_CAP = 8


def _platform_budgets() -> dict:
    """补上没设过的护栏默认值，返回**实际生效**的三元组（进决策日志）。"""
    for k, v in _PLATFORM_DEFAULTS:
        if not str(os.environ.get(k, "") or "").strip():
            os.environ[k] = v
    return {k: os.environ.get(k, "") for k, _ in _PLATFORM_DEFAULTS}


# ======================================================================
# case 级扇出（用户 m42543：「判题机将同时提供所有样例并且允许并发」）
# ======================================================================
# 为什么不是 `-workers`：它只并行**阶段一**（`stage1.pipeline.run_cases` 的进程池）。
# 9 张官方图实测：串行 50.6 s → `-workers 8` 45.1 s（只快 ~10%），且
# `final_results/*_Pred.json` **9/9 逐字节相同**（并发只改速度、不改内容）。
# 交付口径的墙钟分布（官方 case 1，`work\_fast_out_tm_vlm1\final_results\run.log`）：
#   阶段一 3.7 s ｜ 阶段二 0.6 s ｜ 阶段三 0.1 s ｜ VLM 探测+补强 16.3 s ｜ 融合(LLM 轮) 5.8 s
#   = **26.6 s/张**；派生大图（元件多 ⇒ VLM 请求多）实测约 **3.7 min/张**
# ⇒ 50 张串行约 3 h，顶穿「2 小时」硬约束；而阶段二/三/融合层**都是逐 case 顺序循环**。
# 做法：把「一张图的全流程」当不可分割的任务，N 个**独立进程**并行、每个独占一张卡；
# 父进程只做分派 / 看门狗 / 合并 / 兜底（空壳 + decision_log）。
# 用进程而不是线程：CRNN、fusion 的 LLM 轮、以及各模块的模块级状态（`cvparts.hypo.LEDGER`、
# `structmatch.HIST_*`、stage1 的 VLM 熔断计数）都不是线程安全的；进程级隔离还顺带把
# `runs/fusion/_fusion_report.json` 的读-改-写竞态消掉了（每个 lane 一份，父进程再合并）。
_CASE_ARGV_MAP = (
    ("imgsz", "-imgsz"), ("texts", "-texts"), ("mode", "-mode"),
    ("llm_batch", "-llm-batch"),
    ("vlm", "-vlm"), ("vlm_workers", "-vlm-workers"), ("vlm_margin", "-vlm-margin"),
    ("vlm_value", "-vlm-value"), ("vlm_value_margin", "-vlm-value-margin"),
    ("vlm_value_batch", "-vlm-value-batch"), ("vlm_value_side", "-vlm-value-side"),
    ("vlm_value_cols", "-vlm-value-cols"),
    ("vlm_name", "-vlm-name"), ("vlm_name_margin", "-vlm-name-margin"),
    ("vlm_pin", "-vlm-pin"), ("vlm_pin_name", "-vlm-pin-name"),
    ("vlm_desig_fallback", "-vlm-desig-fallback"),
    ("vlm_desig_all", "-vlm-desig-all"),
    ("fusion", "-fusion"), ("llm_max_slots", "-llm-max-slots"),
    ("gpu_frac", "-gpu-frac"), ("gpu_headroom", "-gpu-headroom"),
)


def _visible_cuda_devices() -> int:
    try:
        import torch
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:                                               # noqa: BLE001
        return 0


def _tiler(a):
    """阶段一切块参数 `(tile, overlap)`，或 `None`（= 整图，今天的行为）。

    为什么存在：`stage1/detect.py:202` 早就实现了切块 + 跨瓦片全局 NMS，但**从来
    没有接进交付入口**（`submission/run.py` 里此前搜不到 `tile`），于是 val 池里
    4500×2850 / 3450×2280 的图在 `-imgsz 1600` 下被整体缩掉 ~2.8×，小元件与细脚
    文字一起消失（README 口径 `pin.missing=729`）。默认 0 ⇒ 交付行为不变。
    """
    tile = int(getattr(a, "tile", 0) or 0)
    if tile <= 0:
        return None
    ov = int(getattr(a, "overlap", 0) or 0)
    return (tile, ov if ov > 0 else max(64, tile // 8))


def _cpu_lanes(n_cases: int) -> int:
    """无卡机器上的扇出上限：按核数与可用内存取，取不到的按 1（旧行为）。

    实测（本机 24 核 / 63.7 GiB，`-offline`，池页）：单 case 进程峰值 **2.5 GiB**、
    13.69 s/张（串行）vs 11.43 s/张（4 路，8 张批；小批里每 lane 的模型加载 ~45 s
    压掉了并行收益）。220 张推算：串行 ≈ 50 min → 4 路 ≈ 15 min。
    """
    try:
        ncpu = os.cpu_count() or 1
    except Exception:                                               # noqa: BLE001
        ncpu = 1
    lanes = max(1, min(4, ncpu // 6))
    try:
        import psutil
        free_gib = psutil.virtual_memory().available / float(1 << 30)
        lanes = max(1, min(lanes, int(free_gib // 3.0) or 1))
    except Exception:                                               # noqa: BLE001
        pass
    return max(1, min(lanes, n_cases))


def _resolve_case_workers(a, n_cases: int) -> int:
    """扇出进程数。`-case-workers` > `EDA_CASE_WORKERS` > 自动（可见 GPU 数）。"""
    v = int(getattr(a, "case_workers", -1))
    env = (os.environ.get("EDA_CASE_WORKERS", "") or "").strip()
    if v < 0 and env.lstrip("+-").isdigit():
        v = int(env)
    if v == 0:
        return 1
    if v > 0:
        return max(1, min(v, n_cases))
    ndev = _visible_cuda_devices()
    dev_low = str(getattr(a, "device", "") or "").strip().lower()
    if ndev <= 0 or dev_low == "cpu":
        # ★ 无卡机器：以前这里返回 1（逐 case 串行），220 张按实测 13.69 s/张 是 ~50 min，
        #   再叠上融合 LLM 轮就顶到 `EDA_WALL_BUDGET_S=6900` 看门狗、尾部被补空壳。
        #   lane 之间是独立进程、产物已验证逐字节一致（4 张/8 张两批 A/B），所以按核数/
        #   内存扇出是纯收益。`EDA_CASE_WORKERS=1` 仍可强制串行。
        return _cpu_lanes(n_cases)
    if ndev <= 1:
        return 1                       # 单卡：保持交付默认（逐 case 串行）
    return max(1, min(ndev, n_cases))


def _lane_device(base, lane: int) -> str:
    """lane i 独占一张卡（`-device 0` → 0/1/…/N-1 轮转）。"""
    d = str(base or "0").strip()
    n = _visible_cuda_devices()
    if n <= 0:
        return d
    low = d.lower()
    if d.isdigit():
        return str((int(d) + lane) % n)
    if low.startswith("cuda"):
        i = 0
        if ":" in low:
            try:
                i = int(low.split(":", 1)[1])
            except Exception:                                       # noqa: BLE001
                i = 0
        return "cuda:%d" % ((i + lane) % n)
    return d


def _child_argv(a, out_dir: Path, cases: list, device, vlm_workers: int,
                stage1_workers: int) -> list:
    argv = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
            "-input", str(a.input_dir), "-output", str(out_dir),
            "-cases", ",".join(cases), "-device", str(device),
            "-case-workers", "1",
            "-workers", str(stage1_workers)]
    if getattr(a, "no_wires", False):
        argv.append("-no-wires")
    for dest, flag in _CASE_ARGV_MAP:
        val = getattr(a, dest, None)
        if val is None:
            continue
        if dest == "vlm_workers":
            val = vlm_workers
        argv += [flag, str(val)]
    return argv


def _child_concurrency(a, n_lanes: int):
    """lane 内的并发度 `(vlm_workers, stage1_workers)`：整体并发 = lanes × 每 lane 并发。

    端点通常是**单模型**的兼容端点（8 卡那天是外部服务），并发太高只会让它排队并回
    429/5xx（读数靠 `vlm/client.py` 的 `VLM_RETRY` 兜，但兜回来的是时间）。所以这里
    把「全进程同时在飞的 VLM 请求数」钉在 `VLM_INFLIGHT_CAP` 以内。

    ★ 8 是量出来的，不是拍的（`docs/RESULTS.md` §5.61，官方 10 张、`-lean`、本机 1 卡）：
      1 lane × vw 4 = 4 路在飞 → 与基线逐字节相同；
      2 lane × vw 4 = 8 路在飞 → **10/10 逐字节相同**，80.04，墙钟 114 s；
      4 lane × vw 2 = 8 路在飞 → **10/10 逐字节相同**，80.04，墙钟 122 s；
      8 lane × vw 2 = 16 路在飞 → 只有 4/10 相同、**77.25（-2.79）**、墙钟 407 s（3.8×）。
    即拐点在 8 与 16 之间：8 路在飞时端点给的答案逐字节不变，16 路时答案本身变了
    （过载下采样/截断不同），掉的分靠重试救不回来。所以 8 路扇出时每 lane 只留 1 个
    worker（8 在飞），而不是默认的 2。

    lane 数一多同时也把 stage1 的进程内并行让给 lane 之间的并行（`s1 = 1`）。
    """
    vw = int(getattr(a, "vlm_workers", 4))
    s1 = int(getattr(a, "workers", 1))
    if n_lanes > 1:
        vw = max(1, min(vw, VLM_INFLIGHT_CAP // n_lanes))
        s1 = 1
    return vw, s1


def _scale_doc_xy(doc, s):
    """把竞赛 JSON 里的**所有坐标**乘 s（等比缩放，底原点/顶原点都成立）。

    只碰 `components[*].bbox`（4 个数）与任何含 `x`/`y` 的字典（脚点、网线端点），
    也容忍 `[x, y]` 数对；其余字段（名字、类型、成员关系）逐字不动。
    用于 `EDA_CANON_SCALE` 的「归一化后把坐标乘回原图尺寸」。
    """
    if abs(s - 1.0) <= 1e-9:
        return doc

    def _walk(node):
        if isinstance(node, dict):
            if set(node.keys()) == {"x", "y"} and all(
                    isinstance(node[k], (int, float)) for k in ("x", "y")):
                return {"x": float(node["x"]) * s, "y": float(node["y"]) * s}
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            if len(node) == 2 and all(isinstance(v, (int, float)) for v in node):
                return [float(node[0]) * s, float(node[1]) * s]
            return [_walk(v) for v in node]
        return node

    for _c in (doc.get("components") or {}).values():
        bb = _c.get("bbox") if isinstance(_c, dict) else None
        if isinstance(bb, list) and len(bb) == 4 and all(
                isinstance(v, (int, float)) for v in bb):
            _c["bbox"] = [float(v) * s for v in bb]
    if isinstance(doc.get("pins"), dict):
        doc["pins"] = _walk(doc["pins"])
    if isinstance(doc.get("nets"), dict):
        doc["nets"] = _walk(doc["nets"])
    # `__pin_index` 之类辅助索引里也可能有坐标；缩放它对判题无影响，但保持一致更难出 bug
    for _k in ("__pin_index", "__comp_index"):
        if isinstance(doc.get(_k), dict):
            doc[_k] = _walk(doc[_k])
    return doc




def _sanitize_pred(doc):
    """出口不变式：网表成员必须指向**存在**的元件与引脚。

    `work/_checkup.py` 的 C7 在 200 案 × 10 轮里抓到 1 例（`0019` 的 `net_9` 引用
    `U0.3`，而 `pins["U0"]` 只剩 `pin_1`：那个引脚在后续剪枝里被删了，网里的引用成了
    陈旧记录）。判题机这边表现为「网络成员指向不存在的引脚」，只会白送一个假成员。

    只摘悬空引用，其余逐字节不动；摘空了就整条删网（空网不进 tp 也进不了召回）。
    """
    comps = doc.get("components") or {}
    pins = doc.get("pins") or {}
    nets = doc.get("nets") or {}
    if not isinstance(nets, dict):
        return doc, False
    changed = False
    for k in list(nets):
        rec = nets.get(k) or {}
        hg = str(rec.get("hyperGraph") or "")
        if not hg:
            continue
        keep, bad = [], []
        _rw = False
        for tok in hg.strip().strip("()").split(","):
            tok = tok.strip()
            if "." not in tok:
                continue
            c, _, num = tok.rpartition(".")
            # `*` = 「不指具体引脚」的通配成员：只有**没有引脚**的电源/地符号会这么写
            # （`assemble.STAGE2_SYM_VERTEX` 把它们当纯顶点挂进网）。
            # ⚠ 老注释说「判题机对非 box 类型按 `net_pin_collapse_nonbox` 折叠成元件身份，
            #   脚号不参与比较」—— 那是 **反的**：`EvalConfig.net_pin_collapse_nonbox`
            #   默认 **False**（`metrics.py:131`，与官方产物标定一致），`_net_sets` 直接拿
            #   `net_pin_set()` 的 `(元件, 脚号)` 去配对 ⇒ `*` 永远对不上 GT 的 `1`。
            #   13 页主尺上 **133/231 张 GT 网含无名电源符号**，每张都被这一处扣成部分分。
            if num == "*":
                ok = c in comps
            else:
                ok = c in comps and ("pin_" + num) in (pins.get(c) or {})
            (keep if ok else bad).append(c + "." + num)
        if not bad and not _rw:
            continue
        changed = True
        if not keep:
            nets.pop(k, None)
            continue
        rec["hyperGraph"] = "(" + ",".join(keep) + ")"
        if isinstance(rec.get("_refs"), list):
            ks = set(keep)
            rec["_refs"] = [x for x in rec["_refs"]
                            if isinstance(x, (list, tuple)) and len(x) >= 2
                            and "%s.%s" % (x[0], x[1]) in ks]
    return doc, changed


def _strip_internal_keys(final_dir: Path, fp=None) -> int:
    """导出档去掉内部字段：判题机只认 `components/pins/nets`（评审 O-3）。

    `__pin_index` / `_pin_stubs` 是装配与融合阶段的中间索引，官方 GT 里没有这两个键。
    本地官方尺解析器忽略未知键，但严格解析器可能直接报错——出口只留官方三节 + `case`。
    环境开关 `PRED_STRIP_INTERNAL=0` 可保留（A/B 用）。
    """
    if os.environ.get("PRED_STRIP_INTERNAL", "1") == "0":
        return 0
    n = 0
    for f in sorted(final_dir.glob("*_Pred.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        hit = [k for k in ("__pin_index", "_pin_stubs") if k in doc]
        if not hit:
            continue
        for k in hit:
            doc.pop(k, None)
        try:
            _atomic_write_text(f, json.dumps(doc, ensure_ascii=False))
            n += 1
        except Exception as _e:                                      # noqa: BLE001
            if fp is not None:
                _log(fp, "  %-28s 内部字段清理失败（%s）" % (f.name, _e))
    return n


def _drop_phantom_boxes(final_dir: Path, fp=None) -> int:
    """导出档剔除「零脚 box」：没有任何引脚的 IC 框是闭合导线围出的假件。

    IC 框的判定特征是「框边上有引脚」；一个 `type=="box"` 的件若 `pins[comp]` 为空，
    说明框内没有任何引脚证据，只是被网线围出来的矩形（自制数据里 `C` 与 `R` 并联的
    引线闭环就曾派生出 `U1 Name="220nl"` 这种件）。GT 侧不存在零脚 IC：13 页交付预测
    31 个 box 里零脚 0 个，另 30 张池回归预测里 24 个零脚件删掉后 comp F1 +0.0027。
    环境开关 `PRED_DROP_EMPTY_BOX=0` 关闭（A/B 用）。
    """
    if os.environ.get("PRED_DROP_EMPTY_BOX", "1") == "0":
        return 0
    n = 0
    for f in sorted(final_dir.glob("*_Pred.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        comps = doc.get("components") or {}
        pins = doc.get("pins") or {}
        used = set()
        for rec in (doc.get("nets") or {}).values():
            for item in (rec.get("hyperGraph") or "").strip("()").split(","):
                if "." in item:
                    used.add(item.split(".", 1)[0])
        gone = [k for k, c in comps.items()
                if str(c.get("type") or "").lower() == "box"
                and not (pins.get(k) or {}) and k not in used]
        if not gone:
            continue
        for k in gone:
            comps.pop(k, None)
        n += len(gone)
        try:
            _atomic_write_text(f, json.dumps(doc, ensure_ascii=False))
        except Exception as _e:                                      # noqa: BLE001
            if fp is not None:
                _log(fp, "  %-28s 零脚 box 清理失败（%s）" % (f.name, _e))
    return n


def _publish_one(lane_file: Path, dst: Path, state: dict, stage_dir: Path) -> bool:
    """把 lane 里的一份 `<case>_Pred.json` **升级**进主输出目录（§5.54）。

    ★ 为什么不是「只补缺不覆盖」：`_run()` 的**渐进导出**（阶段二那一段，用户 m42543
    的灾难现场加固）会在阶段二刚跑完时就把**还没融合**的文档写进 lane 的
    `final_results/`，然后阶段三 / 融合层再覆盖它。父进程若「只补缺不覆盖」，抢到的
    第一份（stage2 档）就**永久胜出**：8 路扇出单卡实测 9/10 张交付的是缺
    `__pin_index`、没过融合层的 stage2 文档（13 293 B），而 lane 里躺着的、以及
    `runs/fusion/<case>.json` 才是对的（14 812 B）。

    正确语义 = **按源文件的新旧升级**：lane 里有更新版本才覆盖主目录里的旧版本，
    绝不把更旧的版本写回去。`state` 记「上次发布时**源文件**的 mtime」，所以主目录
    那份（拷贝时刻更新）不会被误判成「更新的版本」。

    拷贝前先 `json.loads` 验一次（子进程可能正在写这个文件），并且**先落同卷暂存
    文件、再 `os.replace`**：判题机任何时刻读到的都是完整 JSON，不会是半截。
    """
    key = lane_file.name
    try:
        st = lane_file.stat()
    except OSError:
        return False
    if dst.exists():
        prev = state.get(key)
        if prev is not None:
            if st.st_mtime_ns <= prev:
                return False
        else:
            # 没有本进程的记忆（例如输出目录里是上一轮留下的旧产物）⇒ 退回「源比它新才升级」
            try:
                if st.st_mtime_ns <= dst.stat().st_mtime_ns:
                    return False
            except OSError:
                return False
    try:
        data = lane_file.read_bytes()
        doc = json.loads(data.decode("utf-8"))
        doc, _changed = _sanitize_pred(doc)
        # `STAGE2_COMP_BOX`：占位框换成字形模板（幂等，关档时返回 0 且逐字节不变）
        if _changed or compbox.apply(doc):
            data = json.dumps(doc, ensure_ascii=False).encode("utf-8")
    except Exception:                                               # noqa: BLE001
        return False                    # 还在写 / 写坏了 ⇒ 下一轮再看
    try:
        tmp = stage_dir / key
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(str(tmp), str(dst))
    except Exception:                                               # noqa: BLE001
        return False
    state[key] = st.st_mtime_ns
    return True


def _merge_lane(lane_dir: Path, final_dir: Path, runs: Path,
                state: dict = None, stage_dir: Path = None) -> int:
    """把一个 lane 的产物并回主输出目录。

    `*_Pred.json` 走 `_publish_one`（**按源文件新旧升级**，见那里的 §5.54 说明）；
    `runs/` 下的中间产物仍旧「只补缺不覆盖」—— 它们只是留档，且同一个 case 的
    `runs/<层>/<case>.json` 在 lane 里也会被后续层覆盖，主目录留第一份无碍。
    """
    n = 0
    state = {} if state is None else state
    stage_dir = (runs / ".pub") if stage_dir is None else stage_dir
    lf = lane_dir / "final_results"
    if lf.is_dir():
        for f in sorted(lf.glob("*_Pred.json")):
            if _publish_one(f, final_dir / f.name, state, stage_dir):
                n += 1
    lr = lane_dir / "runs"
    if lr.is_dir():
        for f in sorted(lr.rglob("*.json")):
            dst = runs / f.relative_to(lr)
            if dst.exists():
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(f), str(dst))
            except Exception:                                       # noqa: BLE001
                pass
    return n


def _publish_preds(lanes: list, final_dir: Path, state: dict = None,
                   stage_dir: Path = None) -> int:
    """把各 lane **已完成**的 `<case>_Pred.json` 立刻补进主输出目录。

    为什么需要：平台侧超时是**硬杀容器**（官方练习环境实测 `Timeout: 1200 seconds
    (20 minutes)`，见 `docs/RESULTS.md` §5.46），父进程的收尾合并根本没机会跑；
    「边跑边发布」保证平台来杀我们时，已经做完的图**仍然躺在输出目录里**。

    每一份都走 `_publish_one`（§5.54：**按源文件新旧升级**）；从前那种「只补缺
    不覆盖」会让阶段二的渐进导出把未融合的旧档永久钉在输出目录里。
    """
    state = {} if state is None else state
    stage_dir = (final_dir.parent / "runs" / ".pub") if stage_dir is None else stage_dir
    n = 0
    for lane in lanes:
        lf = lane["dir"] / "final_results"
        if not lf.is_dir():
            continue
        for f in sorted(lf.glob("*_Pred.json")):
            if _publish_one(f, final_dir / f.name, state, stage_dir):
                n += 1
    return n


def _run_fanout(a) -> int:
    """`-case-workers` > 1 时把 case 分给 N 个独立子进程；否则原样走 `_run(a)`。"""
    in_dir, out_dir = Path(a.input_dir), Path(a.output_dir)
    if not in_dir.is_dir():
        print("输入目录不存在：%s" % in_dir, file=sys.stderr)
        return 2
    items, _dup = _dedupe_items(discover_inputs(in_dir))
    if getattr(a, "cases", ""):
        want = {c.strip() for c in a.cases.split(",") if c.strip()}
        items = [(c, p) for c, p in items if c in want]
    if getattr(a, "limit", 0):
        items = items[:a.limit]
    n_lanes = _resolve_case_workers(a, len(items)) if items else 1
    if n_lanes <= 1:
        return _run(a)

    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = out_dir / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)
    runs = out_dir / "runs"
    lanes_root = out_dir / "_lanes"
    lanes_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    fp = (final_dir / "run.log").open("w", encoding="utf-8")
    _log(fp, "EDA 赛题六 · 提交入口（case 级扇出）")
    _log(fp, "input=%s  output=%s" % (in_dir, out_dir))
    _log(fp, "找到 %d 张图，扇出 %d 个进程（每 lane 独占一张卡）" % (len(items), n_lanes))
    if _dup:
        _log(fp, "⚠️ 输入里 %d 张图与别的图**同名**：%s ⇒ 只保留第一个"
             % (len(_dup), ", ".join("%s <- %s" % (c, p) for c, p in _dup[:6])))

    # lane 内并发度：整体并发 = lanes × 每 lane 并发，别把本地 LLM/VLM 端点打爆。
    # 端点通常是 LM Studio 单模型（127.0.0.1:1234），32 路并发只会让它排队。
    child_vw, child_s1 = _child_concurrency(a, n_lanes)
    ncpu = os.cpu_count() or 4
    child_env = dict(os.environ)
    nthr = str(max(1, ncpu // n_lanes))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        child_env[k] = nthr

    ids = [cid for cid, _p in items]
    lanes = []
    for i in range(n_lanes):
        sub = ids[i::n_lanes]
        if not sub:
            continue
        lane_dir = lanes_root / ("lane%d" % i)
        lane_dir.mkdir(parents=True, exist_ok=True)
        dev = _lane_device(getattr(a, "device", "0"), i)
        argv = _child_argv(a, lane_dir, sub, dev, child_vw, child_s1)
        lf = (lane_dir / "stdout.log").open("w", encoding="utf-8")
        proc = None
        try:
            proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT,
                                    env=child_env)
        except Exception as exc:                                    # noqa: BLE001
            _log(fp, "⚠️ lane%d 起不来（%r）⇒ 这 %d 张只能靠空壳兜底" % (i, exc, len(sub)))
        lanes.append({"i": i, "cases": sub, "dir": lane_dir, "proc": proc,
                      "log": lf, "device": dev})
        _log(fp, "lane%d：device=%s，%d 张，vlm-workers=%d，stdout=%s"
             % (i, dev, len(sub), child_vw, lane_dir / "stdout.log"))

    budget = 0.0
    try:
        budget = float(os.environ.get("EDA_WALL_BUDGET_S", "0") or 0)
    except Exception:                                               # noqa: BLE001
        budget = 0.0
    deadline = (t_start + budget) if budget > 0 else None
    _log(fp, "墙钟预算 %s（EDA_WALL_BUDGET_S；0 = 不限）"
         % ("%.0f s" % budget if budget > 0 else "不限"))
    try:
        pub_every = float(os.environ.get("EDA_PUBLISH_S", "15") or 15)
    except Exception:                                               # noqa: BLE001
        pub_every = 15.0
    if pub_every > 0:
        _log(fp, "增量发布间隔 %.0f s（EDA_PUBLISH_S）：lane 的产物边跑边补进输出目录，"
                 "平台硬杀容器也留得下已完成的图" % pub_every)
    else:
        _log(fp, "增量发布关闭（EDA_PUBLISH_S=0）⇒ 父进程若被平台杀掉则输出目录为空")
    n_pub = 0
    # §5.54：发布状态（记「上次发布时**源文件**的 mtime」，用来按新旧升级）
    #   + 同卷暂存目录（先写它、再 `os.replace`，判题机不会读到半截 JSON）。
    pub_state = {}
    pub_stage = runs / ".pub"
    pending = [ln for ln in lanes if ln["proc"] is not None]
    while pending:
        still = []
        for lane in pending:
            p = lane["proc"]
            if p.poll() is None:
                still.append(lane)
                continue
            lane["rc"] = p.returncode
            try:
                lane["log"].close()
            except Exception:                                       # noqa: BLE001
                pass
            _log(fp, "lane%d 结束 rc=%s" % (lane["i"], lane["rc"]))
        if pub_every > 0 and still:
            got = _publish_preds(lanes, final_dir, pub_state, pub_stage)
            if got:
                n_pub += got
                _log(fp, "增量发布 %d 份（累计 %d）" % (got, n_pub))
        if not still:
            pending = []
            break
        if deadline is not None and time.time() >= deadline:
            for lane in still:
                _log(fp, "⚠️ lane%d 超过墙钟预算 ⇒ 强杀（未产出的图走空壳兜底）" % lane["i"])
                try:
                    lane["proc"].kill()
                    lane["proc"].wait(timeout=30)
                except Exception:                                   # noqa: BLE001
                    pass
                lane["rc"] = -9
                try:
                    lane["log"].close()
                except Exception:                                   # noqa: BLE001
                    pass
            pending = []
            break
        wait = pub_every if pub_every > 0 else 5.0
        if deadline is not None:
            wait = min(wait, max(0.5, deadline - time.time()))
        time.sleep(max(0.2, wait))
    if pub_every > 0:
        got = _publish_preds(lanes, final_dir, pub_state, pub_stage)
        if got:
            n_pub += got
            _log(fp, "增量发布（收尾）%d 份（累计 %d）" % (got, n_pub))

    n_merge = 0
    for lane in lanes:
        n_merge += _merge_lane(lane["dir"], final_dir, runs, pub_state, pub_stage)
    n_shell_here = 0
    for cid, _p in items:
        if not (final_dir / ("%s_Pred.json" % cid)).exists():
            if _write_shell(final_dir, cid, note="扇出 lane 未产出，空壳兜底"):
                n_shell_here += 1
    n_export = len(items) - n_shell_here
    _log(fp, "合并 lane 产物 %d 份；导出 %d 份 <case>_Pred.json%s"
         % (n_merge, n_export,
            ("（另有 %d 份空壳兜底）" % n_shell_here) if n_shell_here else ""))
    _n_phantom = _drop_phantom_boxes(final_dir, fp)
    if _n_phantom:
        _log(fp, "零脚 box 清理：%d 个无引脚 IC 框" % _n_phantom)
    _n_strip = _strip_internal_keys(final_dir, fp)
    if _n_strip:
        _log(fp, "内部字段清理：%d 份 <case>_Pred.json 去掉 __pin_index/_pin_stubs" % _n_strip)

    lane_log = []
    for lane in lanes:
        s = {}
        dl = lane["dir"] / "decision_log.json"
        if dl.exists():
            try:
                s = (json.loads(dl.read_text(encoding="utf-8")) or {}).get("summary") or {}
            except Exception:                                       # noqa: BLE001
                s = {}
        lane_log.append({"lane": lane["i"], "device": lane["device"],
                         "n": len(lane["cases"]), "rc": lane.get("rc"),
                         "cases": lane["cases"], "summary": s})
    cases_log = [{"case": cid, "exported": (final_dir / ("%s_Pred.json" % cid)).exists()}
                 for cid, _p in items]
    wall = round(time.time() - t_start, 1)
    _log(fp, "全部完成，墙钟 %.1fs（%.2f s/张，扇出 %d 路）"
         % (wall, wall / max(1, len(items)), n_lanes))
    fp.close()
    agg = {k: sum(int((l["summary"] or {}).get(k) or 0) for l in lane_log)
           for k in ("n_stage1_ok", "n_stage2_ok", "n_stage3_ok")}
    # 空壳 = lane 自己写的（图坏了 / 该 lane 内部阶段失败）+ 父进程补的
    n_shell_lane = sum(int((l["summary"] or {}).get("n_shell") or 0) for l in lane_log)
    _write_decision(out_dir,
                    {"n_cases": len(items), "n_lanes": n_lanes,
                     "n_stage1_ok": agg["n_stage1_ok"], "n_stage2_ok": agg["n_stage2_ok"],
                     "n_stage3_ok": agg["n_stage3_ok"],
                     "n_exported": n_export, "n_shell": n_shell_lane + n_shell_here,
                     "wall_s": wall,
                     "lanes": lane_log},
                    cases_log, a, t_start, note="case 级扇出")
    print("产物 -> %s" % final_dir)
    return 0


def _run(a) -> int:
    # ★ 在碰 torch / ultralytics 之前把显存上限设好。LLM（LM Studio）和管线
    # 共用一张卡，PyTorch 默认会「能用多少用多少」，把 LLM 挤到 CPU 或自己 OOM。
    if a.gpu_headroom > 0:
        os.environ["EDA_GPU_HEADROOM_MB"] = str(a.gpu_headroom)
    try:
        from gpu_mem import cap_vram, describe as _gpu_describe
        _gpu_info = cap_vram(a.gpu_frac if a.gpu_frac > 0 else None)
    except Exception as _e:                                         # noqa: BLE001
        _gpu_info = "gpu_mem 不可用: %r" % (_e,)

    in_dir, out_dir = Path(a.input_dir), Path(a.output_dir)
    if not in_dir.is_dir():
        print("输入目录不存在：%s" % in_dir, file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = out_dir / "final_results"
    final_dir.mkdir(parents=True, exist_ok=True)
    runs = out_dir / "runs"
    s1_dir, s2_dir, s3_dir = runs / "stage1", runs / "stage2", runs / "stage3"
    s3v_dir = runs / "stage3_vlm"                # VLM 补强后的文档（`-vlm` 开启时才写）
    fuse_dir = runs / "fusion"                   # 融合层产物（`-fusion` 开启时才写）

    log_path = final_dir / "run.log"
    fp = log_path.open("w", encoding="utf-8")
    t_start = time.time()
    _log(fp, "EDA 赛题六 · 提交入口")
    _log(fp, "input=%s  output=%s" % (in_dir, out_dir))
    _log(fp, "显存 " + _gpu_info)
    if a.workers > 1:
        _log(fp, "⚠ workers=%d > 1：每个 stage1 子进程各有一份同样的显存配额，"
                 "并发把总量放大 %d 倍，LLM 会被挤。" % (a.workers, a.workers))

    items = discover_inputs(in_dir)
    items, _dup = _dedupe_items(items)
    if _dup:
        _log(fp, "⚠️ 输入里 %d 张图与别的图**同名**（`<case_name>` 相同）：%s ⇒ 只保留第一个"
                 "（同名无法产出两份独立的 `<case_name>_Pred.json`；`-workers>1` 时"
                 "同名还会并发写同一个 stage1 文件）"
             % (len(_dup), ", ".join("%s <- %s" % (c, p) for c, p in _dup[:6])))
    if a.cases:
        want = {c.strip() for c in a.cases.split(",") if c.strip()}
        items = [(c, p) for c, p in items if c in want]
    if a.limit:
        items = items[:a.limit]
    if not items:
        _log(fp, "没有找到任何图片，结束")
        _write_decision(out_dir, {"n_cases": 0, "n_ok": 0, "n_failed": 0},
                        [], a, t_start, note="no images")
        fp.close()
        return 1
    _log(fp, "找到 %d 张图" % len(items))

    # ------------------------------------------- 页面尺度归一化（`EDA_CANON_SCALE`）
    # 为什么：整条管线的像素常量（文字高度闸 4~34px、`pad_min`、脚外扩、`EDGE_SNAP`、
    #   线掩膜长度…）都是按本数据集的 DPI 调的。同一张图等比放大 1.4× 后官方 10 张
    #   89.79 → **59.6**（`work/_degrade.py` + `EDA_DATA_ROOT`，RESULTS §5.387）。
    #   做法：把每页按「文字高度中位数」缩到标准尺度再跑整条管线，导出时把坐标乘回去。
    #   `EDA_CANON_SCALE=0`（默认）= 完全不动；>0 = 目标文字高度中位数（px）。
    #   `EDA_CANON_DEADZONE`（默认 0.08）：|k-1| 小于它就不缩放，避免把基准集改坏。
    _canon_k = {}
    _canon = float(os.environ.get("EDA_CANON_SCALE", "0") or 0.0)
    if _canon > 0.0:
        _dz = float(os.environ.get("EDA_CANON_DEADZONE", "0.08") or 0.0)
        _lo = float(os.environ.get("EDA_CANON_MIN", "0.5") or 0.5)
        _hi = float(os.environ.get("EDA_CANON_MAX", "2.0") or 2.0)
        _tmp = out_dir / "_canon"
        _tmp.mkdir(parents=True, exist_ok=True)
        try:
            import cv2 as _cv2                                        # noqa: PLC0415
            from stage2.text import detect_texts as _dtexts, imread_unicode as _imr  # noqa: PLC0415
        except Exception as _e:                                       # noqa: BLE001
            _log(fp, "尺度归一化不可用（%s）⇒ 按原尺寸跑" % _e)
            _canon = 0.0
        if _canon > 0.0:
            _new = []
            for _cid, _p in items:
                try:
                    _bgr = _imr(str(_p))
                    _hs = sorted((y1 - y0) for (x0, y0, x1, y1)
                                 in _dtexts(_bgr, min_h=3, max_h=60))
                    _mh = float(_hs[len(_hs) // 2]) if _hs else 0.0
                    _k = min(_hi, max(_lo, _canon / _mh)) if _mh > 0 else 1.0
                except Exception as _e:                               # noqa: BLE001
                    _log(fp, "  %-28s 尺度估计失败（%s）⇒ 原尺寸" % (_cid, _e))
                    _k = 1.0
                if abs(_k - 1.0) <= _dz:
                    _canon_k[_cid] = 1.0
                    _new.append((_cid, _p))
                    continue
                _out = _tmp / ("%s.png" % _cid)
                _bgr2 = _cv2.resize(_bgr, None, fx=_k, fy=_k,
                                    interpolation=_cv2.INTER_AREA if _k < 1.0 else _cv2.INTER_CUBIC)
                _cv2.imwrite(str(_out), _bgr2)
                _canon_k[_cid] = _k
                _new.append((_cid, str(_out)))
                _log(fp, "  %-28s 尺度归一化 k=%.3f（文字高 %s → %.1f，%dx%d → %dx%d）"
                     % (_cid, _k, ("%.0f" % _mh) if _mh else "?", _canon,
                        _bgr.shape[1], _bgr.shape[0], _bgr2.shape[1], _bgr2.shape[0]))
            items = _new

    from stage1.detect import DEFAULT_WEIGHTS
    from stage1.pipeline import run_cases
    _assets = assets.describe()
    _log(fp, "权重 " + json.dumps(_assets, ensure_ascii=False))
    # ★ 加固：资产缺失**当时就喊**。三个资产里有任何一个不在，对应通道会静默降级
    #   （YOLO 权重缺失 = stage1 全灭；CRNN 缺失 = 位号/值全空但产物齐全；
    #   元件库缺失 = 引脚补号/改名白丢实测 +1.60）——「能跑出分但少一大截」最难查。
    for _k, _v in (_assets or {}).items():
        if isinstance(_v, dict) and _v.get("exists") is False:
            _log(fp, "⚠️ 资产缺失：%s = %s —— 该通道会静默降级，"
                     "请检查镜像里是否带上了这个文件" % (_k, _v.get("path")))

    # ------------------------------------ VLM 预探测 + 竖直脚旋转重读的自动兜底
    # ★ 用户 m42543「要考虑各种灾难性的现场」：交付默认 `-vlm auto` ⇒ 端点不可用时
    #   **整个引脚通道不跑**（回到纯 CRNN）。这时 `STAGE2_PINROT`（竖直脚旋转重读）是
    #   唯一的救命通道：官方 10GT 里 case 6 的引脚在无 VLM 口径是**整张 0 分**
    #   （`fast.py --cases 6 --no-vlm` = 53.20 / pin 0.0000），开了它 +2.86；case 7
    #   +3.85、case 10 +2.17（9 张均值 63.02 → 63.69，**7 张逐位不变、无一倒退**）。
    #   反过来「VLM 引脚通道开着」时旋转重读是**净负**（交付口径 case 6 59.94 → 59.67，
    #   引脚 +0.0033 但网络 −0.0101：修好的引脚名会参与电源轨挂标签）⇒ 只在引脚通道
    #   不会跑时自动打开。显式设了 `STAGE2_PINROT` 的人优先，不覆盖。
    vlm_ok = (a.vlm == "on")
    _vlm_why = "显式 -vlm on" if a.vlm == "on" else ""
    if a.vlm == "auto":
        try:
            from vlm import client as _vc
            _t0 = time.time()
            # ★ 用 `preflight(strict=False)` 而不是 `probe()`（2026-10-27，§5.103）：
            #   `probe()` 只回 bool，日志里一律写成「不支持图片」，而**鉴权失败（401）**
            #   与「模型不吃图（400）」在这句话里长得一样 —— token 轮换那天整跑真的
            #   按「不支持图片」记了一笔，排查方向被带偏。detail 里才有 401 / 400 / EMPTY。
            vlm_ok, _why = _vc.preflight(strict=False)
            _why = str(_why or "").replace("\n", " ")[:220]
            _vlm_why = _why or "预探测回 False（无 detail）"
            _log(fp, "VLM 端点预探测（%s，%.1fs，key 来源 %s）：%s"
                 % (_vc.BASE, time.time() - _t0, getattr(_vc, "KEY_SOURCE", "?"),
                    "可用" if vlm_ok else "不可用 ⇒ 跳过补强｜%s" % _why))
            if not vlm_ok and any(t in _why for t in ("401", "Unauthorized", "header")):
                _log(fp, "  ⚠️ 这是**鉴权**问题，不是「模型不支持图片」：设 LLM_API_KEY，"
                         "或把 LM Studio 的 token 写进 eda-pcb/.lmstudio_key（见 RESULTS §5.103）")
        except Exception as _e:                                     # noqa: BLE001
            vlm_ok = False
            _log(fp, "VLM 预探测失败（%s: %s）⇒ 按不可用处理"
                 % (type(_e).__name__, _e))
    if not (vlm_ok and a.vlm_pin == "on") and "STAGE2_PINROT" not in os.environ:
        os.environ["STAGE2_PINROT"] = "1"
        _log(fp, "VLM 引脚通道不会跑 ⇒ 打开 STAGE2_PINROT=1（竖直脚旋转重读；"
                 "无 VLM 口径官方 9 张 +0.67 且九张零回归）")

    # ---------------------------------------------------------- 阶段一
    cases = [Case(split="input", case_id=cid, png_path=str(p), json_path=None)
             for cid, p in items]
    _log(fp, "=== 阶段一：视觉（YOLO + CV + 骨架） %d 张 ===" % len(cases))
    t0 = time.time()
    r1 = run_cases(cases, s1_dir, workers=a.workers, weights=DEFAULT_WEIGHTS,
                   imgsz=a.imgsz, conf=float(os.environ.get("SUBMISSION_CONF", "0.25") or 0.25),
                   device=a.device,
                   use_cv=True, do_wires=not a.no_wires,
                   tiler=_tiler(a))
    ok1 = {r["case"] for r in r1 if r["ok"]}
    _log(fp, "阶段一完成 %d/%d，用时 %.1fs" % (len(ok1), len(r1), time.time() - t0))
    stage1_report = {r["case"]: r for r in r1}

    # ---------------------------------------------------------- 阶段二
    from stage2.run import process as s2_process
    from stage2.text import TextEngine
    _log(fp, "=== 阶段二：文字识别 + 装配 ===")
    engine = TextEngine(device=a.device)
    _log(fp, "识别器 " + engine.describe())
    s2_rows = []
    n_prog = 0
    t0 = time.time()
    for i, cid in enumerate([c.case_id for c in cases if c.case_id in ok1], 1):
        try:
            r = s2_process(cid, s1_dir, s2_dir, engine, texts=a.texts)
            s2_rows.append(r)
            _log(fp, "  [%d] %-28s 元件 %-4d 位号 %-4d %.1fs"
                 % (i, cid, r["n_components"], r["n_designator"], r["t_total"]))
            # ★ 渐进导出（灾难现场加固，用户 m42543）：阶段三是 LLM（~30 s/张）、
            #   VLM 补强 **~3.7 min/张**，19 张交付口径就要 55 min，220 张数小时；
            #   被判题机按墙钟硬杀时**只有已经落进 `final_results/` 的才有分**。
            #   这里每完成一张先把阶段二文档顶上去，后面各层照旧覆盖（导出优先序不变，
            #   分数与非渐进路径逐位相同）⇒ 任何时刻被杀都不至于整批空手。
            try:
                # ★ 2026-11：单进程档原来在这里**直接 copyfile**，绕过了出口不变式
                #   `_sanitize_pred`（lane 档走 `_publish_one` 才会过）⇒ 悬空引用和
                #   `pNNN.*` 通配脚一路写进交付。这里补上：内容没变仍逐字节照抄。
                _src = s2_dir / ("%s.json" % cid)
                _dst = final_dir / ("%s_Pred.json" % cid)
                _doc, _ch = _sanitize_pred(json.loads(_src.read_text(encoding="utf-8")))
                if _ch:
                    _dst.write_text(json.dumps(_doc, ensure_ascii=False), encoding="utf-8")
                else:
                    shutil.copyfile(_src, _dst)
                n_prog += 1
            except Exception as _e:                               # noqa: BLE001
                _log(fp, "  ⚠️ 渐进导出失败 %s: %s: %s"
                     % (cid, type(_e).__name__, _e))
        except Exception as e:                                    # noqa: BLE001
            _log(fp, "  [%d] %-28s 阶段二失败 %s: %s" % (i, cid, type(e).__name__, e))
            fp.write(traceback.format_exc()[-800:] + "\n")
    _log(fp, "阶段二完成 %d 张，用时 %.1fs（渐进导出 %d 份）"
         % (len(s2_rows), time.time() - t0, n_prog))

    # ---------------------------------------------------------- 阶段三
    from stage3.run import process as s3_process
    from stage1 import llm as llm_mod
    mode = a.mode
    if mode == "auto":
        mode = "llm" if llm_mod.available() else "off"
    _log(fp, "=== 阶段三：位号仲裁（mode=%s） ===" % mode)
    if mode == "llm":
        _log(fp, "  LLM " + llm_mod.describe())
    s3_rows = []
    t0 = time.time()
    for i, cid in enumerate([r["case"] for r in s2_rows], 1):
        try:
            r = s3_process(cid, s1_dir, s2_dir, s3_dir, mode=mode,
                           batch=a.llm_batch, max_slots=a.llm_max_slots)
            s3_rows.append(r)
            _log(fp, "  [%d] %-28s 槽位 %-4d 仲裁 %-4d %.1fs"
                 % (i, cid, r["n_slots_total"], r.get("arb", {}).get("n_decided", 0), r["t"]))
        except Exception as e:                                    # noqa: BLE001
            _log(fp, "  [%d] %-28s 阶段三失败 %s: %s" % (i, cid, type(e).__name__, e))
            fp.write(traceback.format_exc()[-800:] + "\n")
    _log(fp, "阶段三完成 %d 张，用时 %.1fs" % (len(s3_rows), time.time() - t0))

    # ------------------------------------- VLM 补强（IC 位号 + 引脚号/名，最后一道）
    # 必须放在**阶段三之后**：阶段三不是拷贝 stage2 的产物，而是从 `_assigned`
    # 结合 stage1 重建文档，放在阶段二之后会被整个冲掉（踩过一次）。
    # 它只改 `components`（位号）/ `pins`（引脚号与名）/ `nets.hyperGraph`，
    # 正好是最终文档的 schema。
    #
    # 220 张官方公开集实测 +0.565（23.9559 -> 24.5211）：component 几乎不动，
    # 增益全部来自引脚（`pin` F1 0.0474 -> 0.0692、tp 69 -> 166）。
    # 为什么位号与引脚必须一起做：`pin_f1` 的 `flat()` 只统计键命中 **GT** box
    # 元件的预测引脚，而我们 box 键命中率只有 0.309 —— 只补引脚会让 tp 69 -> 54、
    # 总分倒退。
    vlm_stats = None
    vlm_err = None
    if a.vlm != "off":
        # ★ 2026-07 审计修复：这一段原来**没有任何 try/except**，而
        # `vlm\enrich.py` 的 `enrich_dir` 一开头就调 `client.preflight()`（strict、
        # 不吞异常）—— 它在设计上就会抛。抛出的那一刻，下面的「导出 final_results」
        # 与「决策日志」都还没跑 ⇒ `final_results/` 里一份 `*_Pred.json` 都没有、
        # 也没有官方要求的 `decision_log.json`，**整批 0 分**。而同一个开关在
        # `-vlm auto` 下只是静默跳过补强（−0.565）。同一开关两种失败模式，最坏的那种
        # 不带任何兜底，必须包起来。
        try:
            from vlm import client as vlm_client
            from vlm import enrich as vlm_enrich
            on = a.vlm == "on"
            if not on:
                # 复用上面的预探测结果（同一次运行只探一次；`STAGE2_PINROT` 的自动开关
                # 也依赖它 —— 两边必须一致，否则会出现「按 VLM 可用关掉旋转重读，
                # 补强却因为端点挂了没跑」的空档）。
                on = vlm_ok
                _log(fp, "VLM 端点探测（%s，复用预探测结果）：%s"
                     % (vlm_client.BASE,
                        "可用" if on else "不可用，跳过补强｜%s" % (_vlm_why or "未探")))
            if on:
                png_of = {cid: str(p) for cid, p in items}
                s3v = s3v_dir
                _log(fp, "=== VLM 补强：%d 张，%d 并发 ===" % (len(ok1), a.vlm_workers))
                t0 = time.time()
                vlm_stats = vlm_enrich.enrich_dir(
                    s3_dir, s3v, png_of, workers=a.vlm_workers,
                    opts={"desig_margin": a.vlm_margin,
                          "value": (a.vlm_value == "on"), "value_margin": a.vlm_value_margin,
                          "value_batch": a.vlm_value_batch, "value_side": a.vlm_value_side,
                          "value_cols": a.vlm_value_cols,
                          "name": (a.vlm_name == "on"), "name_margin": a.vlm_name_margin,
                          "pin": (a.vlm_pin == "on"),
                          "desig_fallback": (a.vlm_desig_fallback == "on"),
                          "desig_all": (a.vlm_desig_all == "on")},
                    cases=[c.case_id for c in cases if c.case_id in ok1])
                s3_dir = s3v                       # 补强后的目录成为导出源
                _log(fp, "VLM 补强完成：补强 %d 张、失败 %d 张｜改名 %d 个元件｜位号调用 %d 次"
                         "｜网格调用 %d 次｜改写引脚 %d 个｜写值 %d/%d 个｜写型号 %d/%d 个"
                         "（型号通道 ERR %d 次）｜%.1fs"
                     % (vlm_stats.get("n", 0), vlm_stats.get("n_failed", 0),
                        vlm_stats["renamed"], vlm_stats["desig_calls"],
                        vlm_stats["grid_calls"], vlm_stats["pins"],
                        vlm_stats.get("value_set", 0), vlm_stats.get("value_calls", 0),
                        vlm_stats.get("name_set", 0), vlm_stats.get("name_calls", 0),
                        vlm_stats.get("name_err", 0),
                        time.time() - t0))
                if vlm_stats.get("n_failed"):
                    _log(fp, "⚠️ 有 %d 张案例的 VLM 补强失败、产物是**未补强的原样拷贝**"
                             "（导出阶段发现不了，只能靠这一行）" % vlm_stats["n_failed"])
                # ★ 护栏可见性：预算到点后 `ask()` 会静默降级成纯 CV（产物照出、只是
                #   没有 VLM 补强），不给一行日志的话没人知道质量为什么掉了。
                _bs = vlm_client.stats()
                _tk = _bs["tokens"]
                # ★ token 用量必须**永远**打出来（不只是预算到点）：赛题
                #   `Score_Token` 是排名项（0.05·(m/M)，消耗越少越高），看不到用量就
                #   没法做「少用外部模型」的取舍。见 `docs/COMPETITION_RULES.md`。
                _bd = ("预算 %.0f/%.0f s｜超预算被拒 %d 次"
                       % (_bs["used_s"], _bs["budget_s"], _bs["refused"])
                       if _bs["budget_s"] > 0 else "预算 0 = 不限")
                _log(fp, "外部推理：发出 %d 次｜失败 %d 次｜%s"
                         "｜空答案 %d 次｜截断 %d 次｜重试 %d 次"
                         "｜token 入 %d / 出 %d / 合计 %d%s"
                     % (_bs["calls"], _bs["errors"], _bd,
                        _bs.get("empty", 0), _bs.get("length", 0), _bs.get("retries", 0),
                        _tk["prompt"], _tk["completion"], _tk["total"],
                        "（⚠️ 已降级成纯 CV：后续调用不再发出）"
                        if _bs["exhausted"] else ""))
            else:
                # 2026-07：这里原来是 `elif a.vlm == "on":`，挂在 `if a.vlm != "off"` 的
                # else 分支里 —— `a.vlm == "on"` 必然满足 `!= "off"`，所以那条警告
                # **永远打不出来**；真正需要告警的「auto 探测失败 ⇒ 静默不补强」反而无声。
                _log(fp, "警告：-vlm %s 但补强未执行（端点不可用），导出的产物**未经 VLM 补强**"
                     % a.vlm)
        except Exception as exc:                    # noqa: BLE001
            # ⚠ 这里原来有一行 `import traceback`。它是**函数内的局部导入**，会让
            # `main()` 整个函数体里的 `traceback` 都变成局部名 —— 于是后面融合层那两个
            # `except` 支（`:401` / `:407` 的 `fp.write(traceback.format_exc()...)`）在
            # **这一行还没执行到**的路径上抛
            # `UnboundLocalError: cannot access local variable 'traceback' where it is
            #  not associated with a value`，把真正的 traceback 整个吃掉。
            # 模块顶层已有 `import traceback`（`:39`），这里不需要再导一次。
            traceback.print_exc()
            vlm_err = "%s: %s" % (type(exc).__name__, exc)
            _log(fp, "⚠️ VLM 补强整段失败：%s —— 继续用**未补强**的阶段三产物导出" % vlm_err)
    a.vlm_stats = vlm_stats                     # 交给 decision_log 记录

    # ---------------------------------------------------------- 融合层
    # 21 节点架构的落地：把 stage1/stage2 的**证据**（OCR 读数、VLM 补强、几何、
    # 网络连通）汇成一个候选池，用「先验 → 解 → 合理性检查」重新定一次值。
    #
    # 它**以 VLM 补强后的文档为底打补丁**（`fusion/run.py:_assemble`），所以在
    # 「补强了」的配置下只会相对补强结果微调，不会退回 stage2。
    #
    # 220 张实测（`work\tools\eval\stage2_eval.py`，均为「真·生产默认档」口径）：
    #   不融合（VLM 补强后的阶段三）                        28.4133
    #   融合 · LLM 轮关（本块，`--no-llm`）                 28.4973   ← 采用
    #   融合 · LLM 轮开（judge + audit）                    28.4200
    # ⇒ 修掉 `priors.py` 的「并列窗口盖过读数」bug 之后，LLM 轮**反而掉分**
    #   （24 张定点档：25.4434 → 25.2688），因为 LLM 轮原先的活就是替那条
    #   先验擦屁股。所以默认保持**LLM 轮关**；`-fusion llm` 是留给实验的。
    # 该 bug 的症状：`0037` 的 `C1→C11`、`0053` 的 `D1→D11/R2→R12/R3→R13`、
    # `0054` 的 `C1→C11`（读数分是负的，先验 `× 0.90` 把它抬了上去）。
    fusion_rows = []
    if a.fusion != "off":
        os.environ["FUSION_RUN_DIR"] = str(runs)
        os.environ["FUSION_OUT_DIR"] = str(fuse_dir)
        # 只有真的补强过才有「VLM 文档」；`-vlm off` 时留一个不存在的目录，
        # 让 `run.py` 走「没有 VLM 文档 ⇒ 全量重装配」那条路。
        os.environ["FUSION_VLM_DIR"] = str(s3v_dir if vlm_stats else runs / "_no_vlm_")
        # LLM 轮（judge/rescan/audit）—— owner m25661：「本来最后的框架就是要让 LLM
        # 参与的，你现在不放进去以后超时了根本无从优化，必须放进去」。
        # 所以 `-fusion on` 现在**默认带 LLM 轮**（`fusion\run.py` 的
        # `DO_LLM = env_on("FUSION_LLM","1")`）：`on` 不覆盖 `FUSION_LLM`，
        # 想临时关掉 LLM 轮但保留融合层就 `FUSION_LLM=0`；`-fusion off` 才整层关，
        # `-fusion llm` 额外打开 cropfix/rescan。
        #
        # ★ 成本实测（10-08，LM Studio 实测在 127.0.0.1:1234，case 1+10，`fast.py`）：
        #   融合层墙钟 **LLM 轮关 3.5 s → 开 62.7 s**（每张 **+29.6 s**；
        #   220 张折算 **+1.8 h**，会与「所有案例 < 2 小时」的硬约束抢预算）。
        #   ⚠ 同一对臂的 `final_results/*_Pred.json` 与 `runs/fusion/_assigned/*.json`
        #   **MD5 逐字节相同** ⇒ 这两张图上 LLM 轮**一个字段都没改**。
        #   归档 `work\_fast_base\{llmab_off,llmab_on}.json`、日志
        #   `work\_fast_out_llmab_{off,on}\final_results\run.log`。
        #   即「省时间」的开关是安全可用的；但**是否默认开**是 owner m25661 的
        #   决定（见上），不要因为一次 A/B 就翻回去。
        os.environ["FUSION_CROPFIX"] = "1" if a.fusion == "llm" else "0"
        # ★ 引脚通道的**「名」那一半**（`fusion/claims.py` 的 `_add_vlm` (d) 段）。
        #   号那一半由 `-vlm-pin` 走 enrich 的 opts（`pin=True` + 出处标记 `_pin_channel`），
        #   这一半是融合层侧的开关，只能用环境变量；`from fusion import run` 在下面，
        #   模块级常量尚未求值 ⇒ 在这里 set 是有效的。
        #   默认 on（全开 +2.88）；评测集若更像 JLC 派生的那 200 张就 `-vlm-pin-name off`
        #   （那时只补号更稳：官方 9 张 +1.95 且零回归）。
        os.environ["FUSION_VLM_PIN_NAME"] = "1" if a.vlm_pin_name == "on" else "0"
        if a.fusion == "off":
            os.environ["FUSION_LLM"] = "0"
        elif a.fusion == "llm":
            os.environ["FUSION_LLM"] = "1"
        _do_llm = os.environ.get("FUSION_LLM", "1").strip().lower() \
            not in ("0", "off", "no", "false", "none", "")
        try:
            from fusion import run as fusion_run
            want = [r["case"] for r in s2_rows]
            _log(fp, "=== 融合层：先验 + 解 + 合理性检查（%d 张，LLM 轮 %s，"
                     "cropfix %s，FUSION_LLM=%s，引脚通道 %s/名 %s） ==="
                 % (len(want), "开" if _do_llm else "关",
                    "开" if a.fusion == "llm" else "关",
                    os.environ.get("FUSION_LLM", "(未设→默认 开)"),
                    "开" if a.vlm_pin == "on" else "关",
                    "开" if a.vlm_pin_name == "on" else "关"))
            t0 = time.time()
            for i, cid in enumerate(want, 1):
                try:
                    fusion_rows.append(
                        fusion_run.run_case(cid, do_llm=_do_llm))
                except Exception as exc:                # noqa: BLE001
                    _log(fp, "  [%d] %-28s 融合失败 %s: %s"
                         % (i, cid, type(exc).__name__, exc))
                    fp.write(traceback.format_exc()[-800:] + "\n")
            _log(fp, "融合完成 %d/%d 张，用时 %.1fs"
                 % (len(fusion_rows), len(want), time.time() - t0))
            # ★ 落 `_fusion_report.json`：`fusion/run.py` 只在它自己的 `main()` 里写这份
            #   报告，而**真实提交路径**（这里）是直接调 `run_case` 的 ⇒ 以前提交产物里
            #   根本没有它，**逐轮明细（judge/audit/rescan 的每一题）事后无从查证**。
            #   合并进已有文件，别覆盖：分批跑（后台作业 / 重跑单张）时覆盖会让报告
            #   只剩最后一批 —— 这个坑 `fusion/run.py:412` 已记过。
            try:
                _rp = fuse_dir / "_fusion_report.json"
                _old = []
                if _rp.exists():
                    _prev = json.loads(_rp.read_text(encoding="utf-8"))
                    _new = {r.get("case") for r in fusion_rows}
                    _old = [x for x in (_prev.get("cases") or [])
                            if x.get("case") not in _new]
                _merged = sorted(_old + fusion_rows,
                                 key=lambda x: str(x.get("case") or ""))
                _atomic_write_text(
                    _rp,
                    json.dumps({"cases": _merged, "run_dir": str(runs)},
                               ensure_ascii=False, indent=1))
                _log(fp, "  逐轮明细 %s（现共 %d 张）"
                     % (_rp.name, len(_merged)))
            except Exception as exc:                    # noqa: BLE001
                _log(fp, "  ⚠️ 写 %s 失败：%s" % ("_fusion_report.json", exc))
        except Exception as exc:                        # noqa: BLE001
            _log(fp, "⚠️ 融合层整段失败：%s: %s —— 继续用未融合的产物导出"
                 % (type(exc).__name__, exc))
            fp.write(traceback.format_exc()[-800:] + "\n")
    a.fusion_stats = {"n": len(fusion_rows), "mode": a.fusion}

    # ---------------------------------------------------------- 导出
    _log(fp, "=== 导出 final_results ===")
    n_export = 0
    n_shell = 0
    exported = set()
    pinzoom_rows = []
    try:
        from vlm import pinzoom as _pzmod                      # noqa: PLC0415
        _pinzoom_on = bool(_pzmod.ON)
    except Exception:                                          # noqa: BLE001
        _pinzoom_on = False
    if _pinzoom_on:
        _log(fp, "引脚名重读通道：开（FUSION_PINZOOM=1）")
    # §5.85：值「紧框重读」通道（`FUSION_VALREAD`，默认 1）。用的是**管线自己的账**
    #   `runs/fusion/_assigned/<case>.json`（回退 `runs/stage2/`）：每颗元件记着它的值
    #   取自哪个文字框 ⇒ 直接拿那个框 + 只问值的 prompt 复读，采信规则见 `vlm/valread.py` 顶注。
    valread_rows = []
    _vrmod = None
    try:
        from vlm import valread as _vrmod                      # noqa: PLC0415
        # ★ 必须挂 `vlm_ok`（= 上面的端点探测结果）：复读通道每槽一次 VLM 调用，端点死了
        #   也不会报错 —— `ask()` 把失败包成 `"ERR:..."` 字符串返回，于是它**照跑到底**、
        #   每槽白等 `VLM_RETRY` 次指数退避。实测（case 3、`LLM_BASE_URL` 指向死端口、
        #   `VLM_TIMEOUT=180`）：预探测 15.2 s 后通道仍然「开」，7 个槽位耗掉 **107 s**
        #   （≈15 s/槽），整张 144 s（正常 18 s）；官方交付 222 张按 ~30 槽/张外推 ≈ 7.8 h
        #   纯浪费，且墙钟被 `EDA_WALL_BUDGET_S` 掐掉时**真的会丢分**。
        _valread_on = bool(_vrmod.ON) and vlm_ok
    except Exception:                                          # noqa: BLE001
        _valread_on = False
    if _valread_on:
        _log(fp, "值重读通道：开（FUSION_VALREAD=1，trigger=%s，guard=%d）"
             % (getattr(_vrmod, "TRIGGER", "?"), int(bool(getattr(_vrmod, "GUARD", 0)))))
    elif bool(getattr(_vrmod, "ON", False)):
        _log(fp, "值重读通道：跳过（VLM 端点不可用 ⇒ 复读只会空转重试）")

    # §5.86：引脚名「紧框复读」通道（`FUSION_PINNAME`，默认 1）。与值通道同构 —— 槽位来自
    #   管线自己的账 `_assigned`（`comps[*].pins[*].name.{box,text,conf}` 就是 CRNN 读名字用的
    #   那个框），紧框 + 只问「这段文字是什么」；采信规则见 `vlm/pinname.py` 顶注（四条闸门，
    #   其中「不与同元件别的脚重名」用来挡 case 4 那种「两个名字框被归到同一根脚」的误采）。
    pinname_rows = []
    _pnmod = None
    try:
        from vlm import pinname as _pnmod                      # noqa: PLC0415
        # ★ 同值通道：必须挂 `vlm_ok`，否则端点死掉时它每槽白等一次重试退避（实测 ≈15 s/槽）。
        _pinname_on = bool(_pnmod.ON) and vlm_ok
    except Exception:                                          # noqa: BLE001
        _pinname_on = False
    if _pinname_on:
        _log(fp, "引脚名复读通道：开（FUSION_PINNAME=1，conf<%.2f 或 w<=%.0f，guard=%d）"
             % (getattr(_pnmod, "CONF", 0.0), getattr(_pnmod, "MAXW", 0.0),
                int(bool(getattr(_pnmod, "GUARD", 0)))))
    elif bool(getattr(_pnmod, "ON", False)):
        _log(fp, "引脚名复读通道：跳过（VLM 端点不可用 ⇒ 复读只会空转重试）")

    # §5.87：元件「同物重复」消重（`FUSION_COMPDEDUP`，默认 **1** = 开；官方 10 张 A/B
    #   81.75 → 81.86、元件 F1 0.9963 → 1.0000）。官方
    #   10 张的 `component.extra` 只剩两条，都是**同一颗符号被检出两次**、其中一次只有假键
    #   `pNNN`（case 4 `p010`(d) 撞 `LED1`；case 10 `p016`(gnd) 撞 `p015`）⇒ 按 bbox 几何删掉
    #   假键那颗；守卫见 `fusion/dedup.py` 顶注（有脚/被网引用的永不删）。
    compdedup_rows = []
    try:
        from fusion import dedup as _cdmod                       # noqa: PLC0415
        _compdedup_on = bool(_cdmod.ON)
    except Exception:                                           # noqa: BLE001
        _compdedup_on = False
    if _compdedup_on:
        _log(fp, "元件消重通道：开（FUSION_COMPDEDUP=1）")

    def _compdedup_step(cid):
        """在**已经落盘**的 `<cid>_Pred.json` 上消重同物元件；失败一律不影响导出。"""
        if not _compdedup_on:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            _st = _cdmod.apply_file(dst_j, dst_j)
            if not _st or not _st.get("dropped"):
                return
            compdedup_rows.append({"case": cid,
                                   **{k: v for k, v in _st.items() if k != "trace"}})
            _log(fp, "  %-28s 元件消重 删 %d：%s"
                 % (cid, _st.get("dropped", 0),
                    "；".join("%s(%s) 与 %s 重复[%s]" % (t.get("drop"), t.get("type"),
                                                        t.get("keep"), t.get("rule"))
                              for t in (_st.get("trace") or []))))
        except Exception as exc:                               # noqa: BLE001
            _log(fp, "  ⚠️ %s 元件消重失败：%s: %s" % (cid, type(exc).__name__, exc))

    # §5.95：网线端点吸附到「电源/地符号」的连接点（`FUSION_PWRCLIP`，默认 **1** = 开；
    #   离线裁定 82.72 → 83.03（+0.31，导线 F1 0.4984 → 0.5291），逐案全部非负）。
    #   病：我们的网线把电源符号自己那截引线也圈进来，比 GT 多画 ~26px（case 2
    #   GT `(933,787)-(1021,787)` vs 我方 `(907,785.81)-(1020,786.52)`）；而 GT 的网线
    #   从符号的连接点起画 —— 连接点 = bbox 四边之一外 10px、居中的点（7 颗 `∅` 全部吻合）。
    #   守卫：端点必须贴在这颗符号上（到 bbox ≤ `fusion/pwrclip.py` 的 `BOXTOL=8`），
    #   否则 T 型接头的中段端点会被误吸。只改 `nets[*].edges` 的端点坐标。
    pwrclip_rows = []
    try:
        from fusion import pwrclip as _pcmod                    # noqa: PLC0415
        _pwrclip_on = bool(_pcmod.ON)
    except Exception:                                           # noqa: BLE001
        _pwrclip_on = False
    if _pwrclip_on:
        _log(fp, "网线端点吸附通道：开（FUSION_PWRCLIP=1）")

    def _pwrclip_step(cid):
        """在**已经落盘**的 `<cid>_Pred.json` 上吸网线端点；失败一律不影响导出。"""
        if not _pwrclip_on:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            _st = _pcmod.apply_file(dst_j, dst_j)
            if not _st or not _st.get("moved"):
                return
            pwrclip_rows.append({"case": cid,
                                 **{k: v for k, v in _st.items() if k != "trace"}})
            _log(fp, "  %-28s 网线吸附 吸 %d 个端点" % (cid, _st.get("moved", 0)))
        except Exception as exc:                               # noqa: BLE001
            _log(fp, "  ⚠️ %s 网线吸附失败：%s: %s" % (cid, type(exc).__name__, exc))

    # ★ `FUSION_EDGECAL`（网线端点偏置校正，默认 0 = 先 A/B）
    #   病：`work/_netl2.py` 逐边归因显示配对网里未配上的 288 fp / 290 fn 中，
    #   端点纵向差在 **+2px 处一个尖峰（118 个）**、超出 310 / 短了 87
    #   ⇒ 上游外推（`stage2/assemble.py:694 EDGE_EXT=20`、`:788 EDGE_EXT_ANY=25`）
    #   让我们的端点整体比 GT 多伸 ~1px。药：两端沿自身轴内收 `TRIM` px。
    #   离线裁定（`work/_edgetrimx.py` + `fast.py --score-only`，官方 10 张）：见 `fusion/edgecal.py` 顶注。
    #   必须排在 `pwrclip` **之后**（吸附会重新移动端点）。
    edgecal_rows = []
    try:
        from fusion import edgecal as _ecmod                    # noqa: PLC0415
        _edgecal_on = bool(_ecmod.ON)
    except Exception:                                           # noqa: BLE001
        _edgecal_on = False
    if _edgecal_on:
        _log(fp, "网线端点偏置校正：开（FUSION_EDGECAL=1，内收 %.1fpx）"
             % float(getattr(_ecmod, "TRIM", 0.0)))

    def _edgecal_step(cid):
        """在**已经落盘**的 `<cid>_Pred.json` 上内收网线两端；失败一律不影响导出。"""
        if not _edgecal_on:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            _st = _ecmod.apply_file(dst_j, dst_j)
            if not _st or not _st.get("moved"):
                return
            edgecal_rows.append({"case": cid,
                                 **{k: v for k, v in _st.items() if k != "trace"}})
            _log(fp, "  %-28s 端点内收 %d 条边" % (cid, _st.get("moved", 0)))
        except Exception as exc:                               # noqa: BLE001
            _log(fp, "  ⚠️ %s 端点内收失败：%s: %s" % (cid, type(exc).__name__, exc))

    # ★ §5.261 `FUSION_PINKEY_FIX`（非法脚号改键，默认 **1** = 开）：见 `_pinkey_step`。
    #   它不改任何几何、不改名字语义，只把 `pin_<非数字>` 收成 `pin_<空闲数字>`，
    #   并把 `hyperGraph` 成员同步改名 —— 契约层的修正，判分上只会去掉 fp。
    pinkey_rows = []
    _pinkey_on = os.environ.get("FUSION_PINKEY_FIX", "1").strip().lower() not in (
        "", "0", "false", "off", "none")
    # ★ `FUSION_PINKEY_ALNUM`（默认 **0** = 关）：把「合法脚号键」从 `^\d{1,3}$` 放宽到
    #   「或 `^[A-Za-z]{1,2}\d{1,3}$`」。动机：连接器（`USB1` 这种 TYPE-C）的脚号本来
    #   就是行字母+列号（`A5`/`A12`/`B12`），GT 的引脚键就是 `pin_A5` 这种形态；旧口径把
    #   它们当非法键改号成 `pin_8/pin_9/…` ⇒ 判分上是 fp+fn 双罚（`open_GT` case11 实测）。
    #   与整关通道（`FUSION_PINKEY_FIX=0`）的区别：那个连 `pin_+1V8`/`pin_U6` 这类真垃圾
    #   键一起放行，本开关只放行「字母+数字」一种形态。
    _pinkey_alnum = os.environ.get("FUSION_PINKEY_ALNUM", "1").strip().lower() not in (
        "", "0", "false", "off", "none")
    _pinkey_ok_re = (re.compile(r"\d{1,3}|[A-Za-z]{1,2}\d{1,3}") if _pinkey_alnum
                     else re.compile(r"\d{1,3}"))
    if _pinkey_on:
        _log(fp, "非法脚号改键通道：开（FUSION_PINKEY_FIX=1），合法键=%s"
             % ("数字或字母数字" if _pinkey_alnum else "仅数字"))

    def _pinkey_step(cid):
        """把**非法脚号键**收成合法数字键（§5.261）；失败一律不影响导出。

        病：`work/_contract.py` 的契约自检在 200 池六块里查出 166 个非法引脚键
        （`pin_+1V8`、`pin_220nF`、`pin_1_2`、`pin_U6`、`pin_C1`），来源是「号」被读成了
        网名/值/型号/BGA 焊盘名。它们是纯 fp，而且会把垃圾成员带进 `hyperGraph`
        （真实输出里出现过 `(U6.U6,...)`、`(U5.+1V8,...)`）。

        口径：只保留 `^\\d{1,3}$` 的脚号；非法键改号到该元件下一个空闲号（原名文本
        若名字空着就写进 `pinname`，信息不丢），并把 `hyperGraph` 里对应成员一起改名 ——
        这样既不丢引脚，也不留非法键。`FUSION_PINKEY_FIX=0` 可关。
        """
        if not _pinkey_on:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        pins = doc.get("pins") or {}
        remap = {}
        n_fix = 0
        for comp, grp in pins.items():
            if not isinstance(grp, dict) or not grp:
                continue
            used = set()
            for pk in grp:
                s = str(pk)[4:] if str(pk).startswith("pin_") else str(pk)
                if re.fullmatch(r"\d{1,3}", s):
                    used.add(int(s))
            nxt = (max(used) + 1) if used else 1
            for pk in list(grp):
                s = str(pk)[4:] if str(pk).startswith("pin_") else str(pk)
                if _pinkey_ok_re.fullmatch(s):
                    continue
                while nxt in used:
                    nxt += 1
                new_pk = "pin_%d" % nxt
                used.add(nxt)
                nxt += 1
                rec = grp.pop(pk)
                if isinstance(rec, dict) and not str(rec.get("pinname") or "").strip():
                    rec["pinname"] = s
                grp[new_pk] = rec
                remap[(str(comp), s)] = str(new_pk)[4:]
                n_fix += 1
        if not n_fix:
            return
        for net in (doc.get("nets") or {}).values():
            hg = net.get("hyperGraph")
            if not isinstance(hg, str) or not (hg.startswith("(") and hg.endswith(")")):
                continue
            mem = [x.strip() for x in hg[1:-1].split(",") if x.strip()]
            out = []
            for x in mem:
                c, _, p = x.partition(".")
                out.append("%s.%s" % (c, remap[(c, p)]) if (c, p) in remap else x)
            net["hyperGraph"] = "(%s)" % ",".join(out)
        _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
        pinkey_rows.append({"case": cid, "rekeyed": n_fix, "map": remap})
        _log(fp, "  %-28s 非法脚号改键 %d（%s）"
             % (cid, n_fix, "；".join("%s.%s→%s" % (c, o, n) for (c, o), n in
                                     list(remap.items())[:6])))

    # ★ §5.285 `FUSION_SYM_YSHIFT`（GND/电源符号的框整体上/下移，默认 0 = 关）
    #   病：`work/_errlist.py` 逐条对账发现，GT 的 gnd 符号框与我们差一个**恒定偏移**
    #   （案例 1 的四个 GND：GT y 43/454/868/…，我们 57/469/882/…，恒定 +14px），
    #   因为我们是拿「GND」这三个字母的文本框当符号框，而真正的地符号画在字的**上方**。
    #   偏移量按 JSON y（向上为正）计：我们偏大 ⇒ 传负值把它压下去。
    _sym_shift = float(os.environ.get("FUSION_SYM_YSHIFT", "0") or 0.0)

    def _symshift_step(cid):
        """把 gnd/v 符号的 bbox（以及它的脚点）整体平移 `FUSION_SYM_YSHIFT` px。"""
        if abs(_sym_shift) < 0.5:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        comps = doc.get("components") or {}
        pins = doc.get("pins") or {}
        moved = []
        for name, rec in comps.items():
            if not isinstance(rec, dict):
                continue
            if str(rec.get("type") or "").lower() not in ("gnd", "v", "vcc"):
                continue
            b = rec.get("bbox")
            if not isinstance(b, (list, tuple)) or len(b) < 4:
                continue
            try:
                b = [float(t) for t in b[:4]]
            except (TypeError, ValueError):
                continue
            b[1] += _sym_shift
            b[3] += _sym_shift
            rec["bbox"] = b
            moved.append(name)
            for pk, pr in (pins.get(name) or {}).items():
                if not isinstance(pr, dict):
                    continue
                if isinstance(pr.get("y"), (int, float)):
                    pr["y"] = float(pr["y"]) + _sym_shift
                pt = pr.get("point")
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        pr["point"] = [float(pt[0]), float(pt[1]) + _sym_shift]
                    except (TypeError, ValueError):
                        pass
        if not moved:
            return
        _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
        _log(fp, "  %-28s 符号框平移 %.1fpx（%d 颗：%s）"
             % (cid, _sym_shift, len(moved), "、".join(moved[:6])))


    #   病二：A→B→C 是一条直线、B 处挂了一条支线时，我们画成 A→C 一条长线，
    #         判分要求两端点各 ≤5px，于是 GT 的 A→B、B→C 两段**都**判错。
    _geo_on = os.environ.get("FUSION_DESEG", "1").strip().lower() not in (
        "", "0", "false", "off", "none")
    _geo_dup = os.environ.get("FUSION_DESEG_DUP", "1").strip().lower() not in (
        "", "0", "false", "off", "none")
    _geo_split = os.environ.get("FUSION_DESEG_SPLIT", "1").strip().lower() not in (
        "", "0", "false", "off", "none")
    _geo_tol = float(os.environ.get("FUSION_DESEG_TOL", "4.0") or 4.0)
    _geo_min = float(os.environ.get("FUSION_DESEG_MINLEN", "6.0") or 6.0)

    # ★ `STAGE2_GND_FILL`（默认 **0 = 关**）：sym16 多族 GND 补充。**只补、不改**。
    #   病：43 类 yolo 对 gnd 的召回在这批图上是 0.892（83 颗漏 9 颗），漏掉的多是
    #   其它工具画的「另一族」符号（细三横 + 竖杆），yolo 给 0.07~0.15 的分数被 0.25 切掉。
    #   16 类小符号模型（`stage1/sym16.py`，imgsz 1280）能看到它们，但直接开 `STAGE1_SYM16`
    #   只有 86.2712（−0.19）：它插在**引脚装配之前**，多出来的 gnd 会把 IC 的脚抢走。
    #   所以放在导出后的**最后一步**（引脚/网都已定稿），只往 `components` 里追加无名 gnd。
    #   闸门与读数（13 页，基線 86.3974；离线扫 = 把候选插进 `_Pred.json` 再打分）：
    #     中心距 12px → 85.8868（−0.51，插 12 个，page5 −3.70 / page11 −1.81 / page12 −1.47）
    #     中心距 16px → 86.4341（+0.04，插 4 个，page12 −0.51）
    #     中心距 20px → **86.4735（+0.0761）**：只插 3 个（case1 的真漏检 ∅323/∅458/∅193），
    #                    逐案只有 case1 +0.99，其余 12 页一分不动；24px 同值。
    #   ⇒ 默认 20px。conf 0.15~0.25 三档同值，取 0.15。
    #   **池门禁（2026-11-05 晚，代码走到 lmJoin 之后）**：13 页 86.9332 → **87.0061（+0.0729）**，逐案只有 case1 +0.95（真漏检 ∅323/∅458/∅193），其余 12 页逐位不动；池 lmJoin 29.6428 → `lmJoinG` 见 §5.583。早先那轮「开这一臂 29.6265」是**基线口径混乱**时期的读数（同一预测摘掉 p9NN 自己就是 29.6292，差 0.22 来自基线不可复现，不是补充造成的），已作废。⇒ **默认 1**。
    _gnd_fill = os.environ.get("STAGE2_GND_FILL", "1").strip().lower() not in (
        "", "0", "false", "off", "none")
    _gnd_conf = float(os.environ.get("STAGE2_GND_FILL_CONF", "0.15") or 0.15)
    _gnd_gap = float(os.environ.get("STAGE2_GND_FILL_GAP", "20") or 20.0)

    _prune_on = os.environ.get("STAGE2_PRUNE_UNKEYED", "1") not in (
        "", "0", "off", "false", "no", "none", "disable", "disabled")
    # ★ 剪掉「无名、且该类型在 GT 里必然具名」的预测件。判分器的坐标匹配只在
    #   **无名 GT ↔ 任意预测件** 之间发生（`metrics.assign_components` 第 2 步
    #   `left_g = [k for k in gc if k not in gt_used and is_unkeyed(k)]`），而 GT 里
    #   只有 `gnd` / `v` / `net_input` 这类符号件会是无名（13 页：gnd 83、v 51、
    #   net_input 8/net_output 1 全具名；池 30 案另有 net_short/other/block 等）。
    #   ⇒ `r/c/l/d/led/switch/testpoint/box` 的无名件**结构上不可能配上**，只是白送
    #   `component.extra`。13 页实测（`work/_pruneorc.py defW2`，池安全版只删 2 颗：
    #   case1 一颗 r、case6 一颗 led）：judge 尺 83.5829 → 83.6295（+0.0466），
    #   pin / netH 两轴逐位不变。
    _prune_types = {"r", "resistor", "c", "capacitor", "l", "inductor", "d", "diode",
                    "led", "switch", "testpoint", "box", "ic", "boxic", "box_ic"}
    _prune_re = re.compile(r"^(p\d+|txt\d+)$")

    _pinfill_on = os.environ.get("STAGE2_PINFILL", "1").strip().lower() not in (
        "", "0", "off", "false", "no", "none", "disable", "disabled")

    def _pinfill_step(cid):
        """库表补脚（`stage2/pinfill.py`，§5.591）：库能查到的 IC/盒体件，按格点补漏检的脚。

        必须在**导出之后**做：stage2 的 `build_doc` 按 stage1 槽点重建 pins，fusion/仲裁
        之后还会再装配一次，所以在 stage2 里补会被覆盖（实测 case6 `U3 pin_1` 补上后又被丢）。
        只增脚，不动 components / nets；无名不补（`name_mismatch` 是 fp+fn 双扣）。
        """
        if not _pinfill_on:
            return
        path = final_dir / ("%s_Pred.json" % cid)
        if not path.exists():
            return
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        try:
            from stage2 import pinfill as _pfm
            st = _pfm.fill(doc)
        except Exception as e:                                  # noqa: BLE001
            _log(fp, "%s 库表补脚失败：%s" % (cid, e))
            return
        if not st.get("added"):
            return
        _atomic_write_text(path, json.dumps(doc, ensure_ascii=False))
        _log(fp, "库表补脚 %d（库命中 %d 件）" % (st["added"], st.get("keys", 0)))

    def _prune_step(cid):
        if not _prune_on:
            return
        path = final_dir / ("%s_Pred.json" % cid)
        if not path.exists():
            return
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            return
        comps = doc.get("components") or {}
        drop = [k for k, v in comps.items()
                if _prune_re.match(str(k))
                and str((v or {}).get("type") or "").strip().lower() in _prune_types]
        if not drop:
            return
        for k in drop:
            comps.pop(k, None)
            for sec in ("pins", "nets"):
                (doc.get(sec) or {}).pop(k, None)
        _atomic_write_text(path, json.dumps(doc, ensure_ascii=False))
        _log(fp, "无名件剪除 %d（%s）" % (len(drop), ",".join(sorted(drop))))


    def _gndfill_step(cid, png):
        """用 sym16 补漏掉的 gnd（add-only：不动任何既有件、不碰 pins/nets）。"""
        if not _gnd_fill:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        try:
            import cv2                                        # noqa: PLC0415
            from stage1 import sym16                           # noqa: PLC0415
        except Exception:                                       # noqa: BLE001
            return
        bgr = cv2.imread(str(png))
        if bgr is None:
            return
        H = int(bgr.shape[0])
        comps = doc.get("components")
        if not isinstance(comps, dict):
            return
        centers = []
        for _k, v in comps.items():
            b = v.get("bbox") if isinstance(v, dict) else None
            if isinstance(b, (list, tuple)) and len(b) == 4:
                centers.append(((float(b[0]) + float(b[2])) / 2.0,
                                H - (float(b[1]) + float(b[3])) / 2.0))
        try:
            dets = sym16.raw_dets(bgr, conf=_gnd_conf)
        except Exception as exc:                                # noqa: BLE001
            _log(fp, "  ⚠️ %s gnd 补充失败：%s: %s" % (cid, type(exc).__name__, exc))
            return
        cands = []
        for d in dets:
            if d.get("cls") != "gnd" or float(d.get("score") or 0) < _gnd_conf:
                continue
            box = [float(t) for t in (d.get("box_img") or [])]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            if any(math.hypot(cx - x, cy - y) < _gnd_gap for x, y in centers):
                continue
            cands.append((float(d["score"]), box))
        cands.sort(key=lambda t: -t[0])
        keep = []
        for sc, box in cands:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            if all(math.hypot(cx - (k[0] + k[2]) / 2.0, cy - (k[1] + k[3]) / 2.0) >= _gnd_gap
                   for _s, k in keep):
                keep.append((sc, box))
        if not keep:
            return
        added, i = [], 0
        for sc, box in keep:
            key = "p9%02d" % i
            while key in comps:
                key += "x"
            i += 1
            comps[key] = {"Name": key, "type": "gnd", "value": "GND",
                          "bbox": [round(box[0], 2), round(H - box[3], 2),
                                   round(box[2], 2), round(H - box[1], 2)],
                          "score": round(sc, 4), "source": "sym16:gnd"}
            added.append(key)
        if added:
            doc["components"] = comps
            _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
            _log(fp, "  %-28s gnd 补充 %d（%s）" % (cid, len(added), "、".join(added[:6])))

    # ★ `STAGE2_TEXT_ANCHOR`（默认 1）：把「位号文字读到了、附近没有元件」的地方补成一颗件。
    #   病：检测器对电感（四段半圆）、轻触开关、输出端口（五边形箭头）三族字形系统性失明，
    #   而它们的位号/标签读得很准 —— case1 `L2` 位号 0.99 距最近件 35.8px（GT 就在旁 7px）、
    #   case11 `SW1` 1.00 / 45.8px、case12 `VOUT` 0.75 / 96px。
    #   做法：候选在 `stage2/text_anchor.py` 里挑（那时手里有全部文字读数），但它进不了最终
    #   文档 —— 融合层会按同框消重吃掉合成件（`_cv_run_ta2` 的 stage2 有 L2、final 没有），
    #   所以 stage2 把候选写到 `runs/stage2/<case>.anchor.json`，由这里在**融合之后**补进去。
    #   上限（`work/_rescueorc.py`，GT 框口径）：三页合计 +2.56（均值 +0.197）；墨迹框 +1.10
    #   —— 差在 case12 的端口墨迹连着导线（121 vs GT 82 ⇒ −0.51），故不补 `net_output`。
    _anchor_on = os.environ.get("STAGE2_TEXT_ANCHOR", "1").strip().lower() not in (
        "", "0", "off", "false")
    #   只放 `l` / `switch` 两类：13 页的上限里它们分别值 +0.33 / +1.27；其余类型在池门禁上
    #   是负的（`anchor` 全开 = 29.5997，逐案 0001 −0.55 全是 5 颗 `TP*`、0011 −0.44 是 3 颗 `D*`；
    #   离线摘掉非 `l/switch` 后回到 29.6344，与 `mergeline` 29.6381 的差在噪声内）。
    # 网标词锚定（STAGE2_TEXT_WORDS）补的 gnd/v/net_input/net_output 也走这条旁路：
    # 键是 p9NN（判题按几何配），框由「导线末端 -> 暗墨迹 -> 类型先验」定，不是模板框。
    _anchor_types = ("l", "switch", "gnd", "v", "net_input", "net_output")

    def _anchor_step(cid):
        if not _anchor_on:
            return
        src = s2_dir / ("%s.anchor.json" % cid)
        if not src.is_file():
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            cand = json.loads(src.read_text(encoding="utf-8")).get("added") or []
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        comps = doc.get("components")
        if not isinstance(comps, dict):
            return
        have = set()
        for k, v in comps.items():
            have.add(str(k))
            if isinstance(v, dict) and v.get("Name"):
                have.add(str(v["Name"]))
        # 交付档的框已经被 compbox 换成字形模板（`led` 14x20 → 53x51），stage2 时算过的
        # 重叠在这里可能才出现（case11 `LED1` 20x20 就压在放大后的 `LED2` 上，−0.95）。
        def _ov(a, b):
            w = min(a[2], b[2]) - max(a[0], b[0])
            h = min(a[3], b[3]) - max(a[1], b[1])
            if w <= 0 or h <= 0:
                return 0.0
            m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            return (w * h / m) if m > 0 else 0.0

        boxes = [list(v["bbox"]) for v in comps.values()
                 if isinstance(v, dict) and isinstance(v.get("bbox"), (list, tuple))
                 and len(v["bbox"]) == 4]
        # 真正写盘前 compbox 还会把占位框换成字形模板（`led` 14x20 → 53x51）。先把那一版
        # 算出来再比重叠，否则 case11 `LED1` 20x20 会压在放大后的 `LED2` 上（实测 −0.95）。
        try:
            _tmp = copy.deepcopy(comps)
            compbox.apply({"components": _tmp})
            boxes = [list(v["bbox"]) for v in _tmp.values()
                     if isinstance(v, dict) and isinstance(v.get("bbox"), (list, tuple))
                     and len(v["bbox"]) == 4]
        except Exception:                                       # noqa: BLE001
            pass
        added = []
        for c in cand:
            key, ty = str(c.get("key") or ""), str(c.get("type") or "")
            bb = c.get("bbox")
            if not key or ty not in _anchor_types:
                continue
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            if key in have:
                continue
            if any(_ov(bb, x) > 0.25 for x in boxes):
                continue
            have.add(key)
            boxes.append(bb)
            comps[key] = {"Name": key, "type": ty, "value": c.get("value"),
                          "bbox": [round(float(v), 2) for v in bb],
                          "score": float(c.get("score") or 0.0),
                          "source": "textanchor"}
            added.append(key)
        if added:
            doc["components"] = comps
            _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
            _log(fp, "  %-28s 文字锚定补件 %d（%s）" % (cid, len(added), "、".join(added[:6])))

    # ★ `STAGE2_DESIG_FILL`（默认 0 = 关）：容灾——「件在、位号没读到」时把 `pNNN` 改成
    #   邻近的孤立位号文字，而不是整颗丢键。侧车由 `stage2/run.py` 的 `desig_candidates()`
    #   写出（`<case>.desig.json`）。三道闸：文字到件框 <= 12px、类型前缀一致、键不冲突。
    #   实测：13 页 87.2015 -> 87.2413（+0.0398），只改 1 件 `case1 p029 led -> LED2`
    #   （文字距离 0.0px、conf 0.947）；池门禁 `dfill` = 29.6510，与 `shape1` 逐位相同 ⇒ 默认 1。
    _desigfill_on = os.environ.get("STAGE2_DESIG_FILL", "1").strip().lower() not in (
        "", "0", "off", "false")

    def _desigfill_step(cid, png=None):
        if not _desigfill_on:
            return
        src = s2_dir / ("%s.desig.json" % cid)
        if not src.is_file():
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            cand = json.loads(src.read_text(encoding="utf-8")).get("reads") or []
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        hgt = None
        try:
            import cv2                                           # noqa: PLC0415
            _ap = str(Path(png).resolve()) if png else None
            img = cv2.imread(_ap) if _ap else None
            if img is not None:
                hgt = int(img.shape[0])
        except Exception:                                       # noqa: BLE001
            hgt = None
        if not hgt:
            return
        # 距离判据要用 compbox 换过的模板框：占位框（`led` 14x20）离位号文字常有 30~40px。
        tmpl = {}
        try:
            _tmpc = copy.deepcopy(doc.get("components") or {})
            compbox.apply({"components": _tmpc})
            tmpl = _tmpc
        except Exception:                                       # noqa: BLE001
            tmpl = {}
        try:
            from stage2 import desig_fill as _dfill        # noqa: PLC0415
            renamed = _dfill.rename_components(doc, cand, hgt, tmpl=tmpl)
        except Exception:                                       # noqa: BLE001
            return
        if renamed:
            _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
            _log(fp, "  %-28s 位号回填 %d（%s）" % (
                cid, len(renamed), "、".join("%s->%s(%.1f)" % r for r in renamed[:6])))

    # ★ `STAGE2_HOLE_TYPE`（默认 1 = 开）：把**小尺寸**的 `H*` 安装孔件从 `box` 改成 `pin`。
    #   病（open_GT case1 `H1`/`H2`）：GT 把这两个 39x20 的圆孔符号记成 `pin`，我们发 `box`
    #   ⇒ `component.type_mismatch`，一对赔 fp+fn；case13 的 `H1` 是 58x116 的连接器体，
    #   GT 记 `box` ⇒ 判据必须看尺寸，只认键名会把 case13 改坏。
    #   读数（final3 基座 86.4049、work\_typefix_orc.py）：无闸门改 3 件 → case1 +1.25 /
    #   case13 −1.97；加尺寸闸门（h<=40 且 w<=60）后只剩 case1 两件 ⇒ **+0.0963**，case13 不动。
    #   13 页实测 `hole1` = 86.5011 / 85.1350 / 82.7407（+0.0962）；池门禁 `holine`（含 LINE_FIX）
    #   = 29.6346 vs final3 29.6292（+0.0054，GPU 噪声 ±0.007 内）⇒ 默认翻 1。
    _hole_type = os.environ.get("STAGE2_HOLE_TYPE", "1").strip().lower() not in (
        "", "0", "off", "false")
    _hole_h, _hole_w = 40.0, 60.0

    def _holetype_step(cid):
        if not _hole_type:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        comps = doc.get("components") or {}
        n = 0
        for k, c in comps.items():
            if not isinstance(c, dict) or c.get("type") != "box":
                continue
            if not re.fullmatch(r"H\d+", str(k)):
                continue
            b = c.get("bbox")
            if not b or len(b) < 4:
                continue
            if (float(b[3]) - float(b[1])) <= _hole_h and (float(b[2]) - float(b[0])) <= _hole_w:
                c["type"] = "pin"
                n += 1
        if n:
            doc["components"] = comps
            _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
            _log(fp, "  %-28s 安装孔类型 %d" % (cid, n))

    def _degeom_step(cid):
        """线段层两步清理（§5.283）：去掉「来回重描」、在真结点处把长线切开。"""
        if not _geo_on:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return

        def _seg_of(raw):
            try:
                return [(float(raw[0]["x"]), float(raw[0]["y"])),
                        (float(raw[1]["x"]), float(raw[1]["y"]))]
            except (KeyError, IndexError, TypeError, ValueError):
                return None

        def _len(s):
            return math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1])

        def _unit(s):
            L = _len(s) or 1.0
            return (s[1][0] - s[0][0]) / L, (s[1][1] - s[0][1]) / L, L

        def _proj(pt, s):
            ux, uy, _ = _unit(s)
            return (pt[0] - s[0][0]) * ux + (pt[1] - s[0][1]) * uy

        def _perp(pt, s):
            ux, uy, _ = _unit(s)
            return abs(-(pt[0] - s[0][0]) * uy + (pt[1] - s[0][1]) * ux)

        def _cover(a, b):
            """b 在 a 上的覆盖率（要求 b 两端垂距都 ≤ tol）。"""
            L = _len(a)
            if L < 1e-6:
                return 0.0
            ts = []
            for p in b:
                if _perp(p, a) > _geo_tol:
                    return 0.0
                ts.append(_proj(p, a))
            return max(0.0, min(L, max(ts)) - max(0.0, min(ts))) / L

        n_dup = n_split = 0
        for net in (doc.get("nets") or {}).values():
            edges = net.get("edges") or {}
            segs = [s for s in (_seg_of(v) for v in edges.values()) if s]
            if len(segs) < 2:
                continue
            # 病一：来回重描 —— 方向近反向、垂距小、重叠大；保留更长的一条
            keep = [True] * len(segs)
            for i in range(len(segs)):
                if not keep[i]:
                    continue
                ui = _unit(segs[i])
                for j in range(i + 1, len(segs)):
                    if not keep[j]:
                        continue
                    uj = _unit(segs[j])
                    if ui[2] < _geo_min or uj[2] < _geo_min:
                        continue
                    if ui[0] * uj[0] + ui[1] * uj[1] > -0.87:     # 夹角 < 150°
                        continue
                    if max(_cover(segs[i], segs[j]), _cover(segs[j], segs[i])) < 0.5:
                        continue
                    if ui[2] >= uj[2]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                    n_dup += 1
            segs = [s for s, k in zip(segs, keep) if k]
            # 病二：T 型结点 —— 别的线段端点落在本线段内部（离两端 ≥ min_len）
            # 只认**方向不同**的端点（真支线）；同向端点多半是同一根线被切过，切它只会增 fp。
            ends = []
            for s in segs:
                ux, uy, _ = _unit(s)
                ends.append((s[0], (ux, uy)))
                ends.append((s[1], (-ux, -uy)))
            out = []
            for s in segs:
                L = _len(s)
                if L < 2 * _geo_min + 1e-6:
                    out.append(s)
                    continue
                ux, uy, _ = _unit(s)
                cuts = set()
                for e, (ex, ey) in ends:
                    if abs(ux * ex + uy * ey) > 0.87:            # 与本体夹角 < 30°，不算支线
                        continue
                    d = _perp(e, s)
                    if d > _geo_tol:
                        continue
                    t = _proj(e, s)
                    if _geo_min <= t <= L - _geo_min:
                        cuts.add(round(t, 3))
                if not cuts:
                    out.append(s)
                    continue
                ux, uy, _ = _unit(s)
                seq = [0.0] + sorted(cuts) + [L]
                for t0, t1 in zip(seq, seq[1:]):
                    if t1 - t0 < 1e-6:
                        continue
                    out.append([(s[0][0] + ux * t0, s[0][1] + uy * t0),
                                (s[0][0] + ux * t1, s[0][1] + uy * t1)])
                n_split += len(seq) - 2
            if len(out) == len(segs):
                continue
            net["edges"] = {"edge_%d" % (k + 1): [
                {"x": round(a[0], 2), "y": round(a[1], 2)},
                {"x": round(b[0], 2), "y": round(b[1], 2)}] for k, (a, b) in enumerate(out)}
        if not (n_dup or n_split):
            return
        _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
        _log(fp, "  %-28s 线段清理：去重描 %d、切结点 %d" % (cid, n_dup, n_split))

    # ★ `STAGE2_LINE_FIX`（默认 1 = 开）：线段层两个小动作——把端点吸附到 2px 内**我们自己的**
    #   引脚点、删掉「两端都落在同网另一条更长边 6px 内」的残段（长度 <40px）。
    #   病（net_line_f1 只认两端都 ≤5px 的一一配对）：我们的网线不以引脚为锚（端点 p50 距引脚
    #   20~50px，GT 一半端点正好在引脚上）、且常多出 10~40px 的残段 ⇒ 只要端点差 13px 就整条不算 tp。
    #   读数（`work\_line_stub2.py hole1`，base 86.5011，line 轴 4.224→4.127）：**+0.0971**，
    #   逐案 1+0.02 4+0.06 5+0.15 6+0.08 7+0.12 9+0.56 10+0.26，**无任何页变差**；
    #   对照甲：按长度一刀切（stub20 + snap2）= +0.1113 但 case8 −0.23（GT 有 33 条 <20px 的真短边）；
    #   对照乙：吸附半径 8/12/16/20px = −0.02/−0.08/−0.21/−0.42（会把端点吸到隔壁脚上）。
    #   管线内实测 `holine2`（HOLE_TYPE=1 + LINE_FIX=1）= **86.5500 / 85.1840 / 82.7880**
    #   （比 hole1 的 86.5011 +0.0489，逐案 1+0.02 5+0.10 9+0.48 10+0.04，无变差）；
    #   池门禁 `holine` = 29.6346 vs final3 29.6292 ⇒ 默认翻 1。
    _line_fix = os.environ.get("STAGE2_LINE_FIX", "1").strip().lower() not in (
        "", "0", "off", "false")
    _lf_r, _lf_tol, _lf_maxlen = 2.0, 6.0, 40.0

    def _pt_seg(pt, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        if L2 <= 1e-9:
            return math.hypot(pt[0] - a[0], pt[1] - a[1])
        t = max(0.0, min(1.0, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / L2))
        return math.hypot(pt[0] - (a[0] + t * dx), pt[1] - (a[1] + t * dy))

    def _linefix_step(cid):
        if not _line_fix:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        pts = []
        for comp in (doc.get("pins") or {}).values():
            for pin in (comp or {}).values():
                p = (pin or {}).get("point")
                if isinstance(p, dict):
                    try:
                        pts.append((float(p["x"]), float(p["y"])))
                    except (KeyError, TypeError, ValueError):
                        pass
        n_snap = n_dup = n_stub = 0
        for net in (doc.get("nets") or {}).values():
            edges = net.get("edges") or {}
            segs = {}
            for k, v in edges.items():
                try:
                    segs[k] = [(float(v[0]["x"]), float(v[0]["y"])),
                               (float(v[1]["x"]), float(v[1]["y"]))]
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
            if not segs:
                continue
            # 一、端点吸附
            for k, s in segs.items():
                new = []
                for pt in s:
                    best = None
                    for q in pts:
                        dd = math.hypot(q[0] - pt[0], q[1] - pt[1])
                        if dd <= _lf_r and (best is None or dd < best[0]):
                            best = (dd, q[0], q[1])
                    if best is None:
                        new.append(pt)
                    else:
                        new.append((best[1], best[2]))
                        n_snap += 1
                segs[k] = new
            # 二、完全重复的边（无序端点，0.5px 内）
            keep = {}
            for k, s in segs.items():
                sig = None
                for k2, s2 in keep.items():
                    if (math.hypot(s[0][0] - s2[0][0], s[0][1] - s2[0][1]) <= 0.5 and
                            math.hypot(s[1][0] - s2[1][0], s[1][1] - s2[1][1]) <= 0.5) or \
                       (math.hypot(s[0][0] - s2[1][0], s[0][1] - s2[1][1]) <= 0.5 and
                            math.hypot(s[1][0] - s2[0][0], s[1][1] - s2[0][1]) <= 0.5):
                        sig = k2
                        break
                if sig is None:
                    keep[k] = s
                else:
                    n_dup += 1
            segs = keep
            # 三、被同网更长边覆盖的残段
            drop = []
            for k, s in segs.items():
                ln = math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1])
                if ln >= _lf_maxlen:
                    continue
                for k2, s2 in segs.items():
                    if k2 == k:
                        continue
                    if math.hypot(s2[1][0] - s2[0][0], s2[1][1] - s2[0][1]) <= ln:
                        continue
                    if (_pt_seg(s[0], s2[0], s2[1]) <= _lf_tol and
                            _pt_seg(s[1], s2[0], s2[1]) <= _lf_tol):
                        drop.append(k)
                        break
            for k in drop:
                segs.pop(k, None)
            n_stub += len(drop)
            if len(segs) != len(edges) or n_snap or n_stub:
                net["edges"] = {"edge_%d" % (i + 1): [
                    {"x": round(a[0], 2), "y": round(a[1], 2)},
                    {"x": round(b[0], 2), "y": round(b[1], 2)}]
                    for i, (a, b) in enumerate(segs.values())}
        if not (n_snap or n_dup or n_stub):
            return
        _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
        _log(fp, "  %-28s 线段修正：吸附 %d、重复 %d、残段 %d"
             % (cid, n_snap, n_dup, n_stub))

    # ★ `STAGE2_LINE_MERGE`（默认 1 = 开）：把同网**共线**碎段并回一条，但**接缝处有分支就保留**
    #   （判据：接缝 5px 内没有第三条边的端点）。
    #   病（case13 `net_3` vs GT `net_1`）：GT 竖线只有一条 `(607,560)-(607,328)`，我们切成 4 段
    #   `384-418 / 413-562 / 327-360 / 356-388`（互相还重叠）⇒ 4 条里最多 1 条能配上，其余全 fp+fn。
    #   读数（`work\_line_junc.py defLT`，base 86.5500）：jun=3 +0.0302、**jun=5 +0.0869**、jun=6 +0.0867、
    #   jun=8 +0.0801、jun=10 +0.0494；jun=5 逐案 1+0.20 3+0.04 4+0.19 5+0.05 7+0.06 9+0.09 10+0.09
    #   12+0.48 13+0.09，只有 6 −0.06。
    #   对照判否：**不看分支点**的共线合并 −0.47（gap0）/ −2.01（gap5），只合并重叠段也 −0.12~−0.29
    #   ⇒ GT 的切分点是「真分支处」，盲合并会丢 tp。
    #   管线读数：merge1 = 86.6369 / 85.2709 / 82.8675（+0.0869 vs defLT 86.5500）；
    #   池门禁 mergeline（GPU，30 案，--pool-ref holine）= 29.6381 vs holine 29.6346（+0.0035，噪声内）
    #   ⇒ 13 页有增益、池上中性，默认翻 1。
    _line_merge = os.environ.get("STAGE2_LINE_MERGE", "1").strip().lower() not in (
        "", "0", "off", "false")
    _lm_perp, _lm_jun = 2.0, 5.0

    def _linemerge_step(cid, _depth=0):
        if not _line_merge:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        n_merge = 0
        for net in (doc.get("nets") or {}).values():
            edges = net.get("edges") or {}
            segs = {}
            for k, v in edges.items():
                try:
                    segs[k] = [(float(v[0]["x"]), float(v[0]["y"])),
                               (float(v[1]["x"]), float(v[1]["y"]))]
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
            if len(segs) < 2:
                continue
            ends = []
            for k, s in segs.items():
                ends.append((k, s[0]))
                ends.append((k, s[1]))

            def _branch(pt, ka, kb):
                for k2, e2 in ends:
                    if k2 in (ka, kb):
                        continue
                    if math.hypot(e2[0] - pt[0], e2[1] - pt[1]) <= _lm_jun:
                        return True
                return False

            groups = []
            for k, s in segs.items():
                dx, dy = s[1][0] - s[0][0], s[1][1] - s[0][1]
                if math.hypot(dx, dy) < 1e-6:
                    continue
                horiz = abs(dx) >= abs(dy)
                if horiz and abs(dy) > _lm_perp + 1e-9:
                    continue
                if (not horiz) and abs(dx) > _lm_perp + 1e-9:
                    continue
                off = (s[0][1] + s[1][1]) / 2.0 if horiz else (s[0][0] + s[1][0]) / 2.0
                t0 = min(s[0][0], s[1][0]) if horiz else min(s[0][1], s[1][1])
                t1 = max(s[0][0], s[1][0]) if horiz else max(s[0][1], s[1][1])
                for g in groups:
                    if g["axis"] != horiz or abs(g["off"] - off) > _lm_perp:
                        continue
                    lo, hi = max(g["t0"], t0), min(g["t1"], t1)
                    if t0 > g["t1"] + 1e-9 or t1 < g["t0"] - 1e-9:
                        continue                     # 有间隙：不并（GT 在此处是真断开）
                    seam = (lo + hi) / 2.0
                    pt = (seam, off) if horiz else (off, seam)
                    if _branch(pt, g["keys"][0], k):
                        continue
                    g["t0"] = min(g["t0"], t0)
                    g["t1"] = max(g["t1"], t1)
                    g["off"] = (g["off"] * g["n"] + off) / (g["n"] + 1)
                    g["n"] += 1
                    g["keys"].append(k)
                    break
                else:
                    groups.append({"axis": horiz, "off": off, "t0": t0, "t1": t1,
                                   "n": 1, "keys": [k]})
            if not any(len(g["keys"]) > 1 for g in groups):
                continue
            out = []
            used = set()
            for g in groups:
                if g["keys"] and len(g["keys"]) > 1:
                    n_merge += len(g["keys"]) - 1
                if g["axis"]:
                    out.append([(g["t0"], g["off"]), (g["t1"], g["off"])])
                else:
                    out.append([(g["off"], g["t0"]), (g["off"], g["t1"])])
                used.update(g["keys"])
            for k, s in segs.items():
                if k not in used:
                    out.append(s)
            net["edges"] = {"edge_%d" % (i + 1): [
                {"x": round(a[0], 2), "y": round(a[1], 2)},
                {"x": round(b[0], 2), "y": round(b[1], 2)}]
                for i, (a, b) in enumerate(out)}
        if not n_merge:
            return
        _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
        _log(fp, "  %-28s 共线合并 %d" % (cid, n_merge))
        if _depth < 2:
            # 单遍合并不收敛：第一遍并完会再暴露出可并的相邻段（离线实测 42->40）。
            _linemerge_step(cid, _depth + 1)

    # ★ `STAGE2_LINE_JOIN`（默认 1 = 开）：把**悬空端点**接到同网另一条边上，接完再清一次残段。
    #   病（case13 `PR net_3`）：GT 竖线 `(607,560)-(607,328)`，我们的竖线停在 y=356，
    #   与上方横线差 28px ⇒ 两端都不在 5px 内，整条边配不上。
    #   判据：端点 3px 内没有别的端点（悬空）+ 到同网另一条边的垂足 <= 12px（垂足须落段内）。
    #   读数（`work\_line_join.py lmIter`，base 86.8286 / line 4.020）：只接边 L=6 +0.0505、
    #   L=10/16 **+0.0671**（逐案 1+0.31 3+0.22 7+0.15 9+0.24 11−0.34 12+0.29）、L=24 +0.0192；
    #   接边后再清残段（tol6/40）**+0.1046**、tol8/40 +0.1066 ⇒ 默认 12px + tol6/40。
    #   落点：**两线交点**（`_lj_isect`）优先、退化才用垂足。原因（case11 −0.34）：垂足会把
    #   `net_4 edge_9` 的竖线端点从 `(101,314)` 拉到 `(104.7,314.4)`，整条边变斜，与 GT `(97,312)`
    #   横差 7.7px > tol 5 ⇒ 丢 tp。改成交点后（`work\_line_join2.py lmIter`）：isect mm0.5
    #   **+0.0992**（1+0.39，11 归零）、再清残段 **86.9654（+0.1368）**；垂足最佳只有 mm4.0 +0.1307。
    _line_join = os.environ.get("STAGE2_LINE_JOIN", "1").strip().lower() not in (
        "", "0", "off", "false")
    _lj_lim, _lj_free = 12.0, 3.0
    _lj_pin = 2.5   # 端点已落在自家引脚上（<=2.5px）就不动：case11 edge_1 曾被拉到 y=341.3（离 GT 332 差 9.3px）


    def _lj_foot(pt, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        ln2 = dx * dx + dy * dy
        if ln2 < 1e-9:
            return None
        t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / ln2
        if t < -1e-6 or t > 1 + 1e-6:
            return None
        return ax + t * dx, ay + t * dy

    def _lj_isect(a1, a2, b1, b2):
        x1, y1 = a1
        x2, y2 = a2
        x3, y3 = b1
        x4, y4 = b2
        den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        if abs(den) < 1e-9:
            return None
        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    def _linejoin_step(cid):
        if not _line_join:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            doc = json.loads(dst_j.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            return
        n_move = n_stub = 0
        _pins = []
        for _c in (doc.get("pins") or {}).values():
            for _p in _c.values():
                _pt = _p.get("point") if isinstance(_p, dict) else None
                if _pt:
                    _pins.append((float(_pt["x"]), float(_pt["y"])))
        for net in (doc.get("nets") or {}).values():
            segs = {}
            for k, v in (net.get("edges") or {}).items():
                try:
                    segs[k] = [(float(v[0]["x"]), float(v[0]["y"])),
                               (float(v[1]["x"]), float(v[1]["y"]))]
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
            if len(segs) < 2:
                continue
            ends = []
            for k, sg in segs.items():
                ends.append((k, sg[0]))
                ends.append((k, sg[1]))
            for k, sg in segs.items():
                for i in (0, 1):
                    pt = sg[i]
                    if any(k2 != k and math.hypot(e[0] - pt[0], e[1] - pt[1]) <= _lj_free
                           for k2, e in ends):
                        continue
                    if _lj_pin and any(math.hypot(p[0] - pt[0], p[1] - pt[1]) <= _lj_pin
                                       for p in _pins):
                        continue
                    best = None
                    for k2, s2 in segs.items():
                        if k2 == k:
                            continue
                        f = _lj_isect(sg[0], sg[1], s2[0], s2[1])
                        if f is None:
                            f = _lj_foot(pt, s2[0], s2[1])
                        if f is None:
                            continue
                        # 横平竖直：只许沿自身方向延长到交点（水平边只动 x、垂直边只动 y）
                        if abs(sg[1][1] - sg[0][1]) <= 1.5:
                            f = (f[0], sg[0][1])
                        elif abs(sg[1][0] - sg[0][0]) <= 1.5:
                            f = (sg[0][0], f[1])
                        d = math.hypot(f[0] - pt[0], f[1] - pt[1])
                        if d <= 0.5 or d > _lj_lim:
                            continue
                        if best is None or d < best[0]:
                            best = (d, f)
                    if best is None:
                        continue
                    sg[i] = best[1]
                    n_move += 1
            drop = []
            for k, sg in segs.items():
                ln = math.hypot(sg[1][0] - sg[0][0], sg[1][1] - sg[0][1])
                if ln >= _lf_maxlen:
                    continue
                for k2, s2 in segs.items():
                    if k2 == k:
                        continue
                    if math.hypot(s2[1][0] - s2[0][0], s2[1][1] - s2[0][1]) <= ln:
                        continue
                    if (_pt_seg(sg[0], s2[0], s2[1]) <= _lf_tol and
                            _pt_seg(sg[1], s2[0], s2[1]) <= _lf_tol):
                        drop.append(k)
                        break
            for k in drop:
                segs.pop(k, None)
            n_stub += len(drop)
            if n_move or drop:
                net["edges"] = {"edge_%d" % (i + 1): [
                    {"x": round(a[0], 2), "y": round(a[1], 2)},
                    {"x": round(b[0], 2), "y": round(b[1], 2)}]
                    for i, (a, b) in enumerate(segs.values())}
        if not (n_move or n_stub):
            return
        _atomic_write_text(dst_j, json.dumps(doc, ensure_ascii=False))
        _log(fp, "  %-28s 端点接边 %d、残段 %d" % (cid, n_move, n_stub))

    def _pinname_step(cid, png):
        """在**已经落盘**的 `<cid>_Pred.json` 上补一层引脚名复读；失败一律不影响导出。"""
        if not _pinname_on:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            _ap = runs / "fusion" / "_assigned" / ("%s.json" % cid)
            if not _ap.exists():
                _ap = runs / "stage2" / "_assigned" / ("%s.json" % cid)
            _st = _pnmod.apply_file(png, _ap, dst_j, dst_j)
            if not _st:
                return
            pinname_rows.append({"case": cid,
                                 **{k: v for k, v in _st.items() if k != "trace"}})
            if _st.get("adopted"):
                _log(fp, "  %-28s 引脚名复读 槽 %d 采用 %d 调用 %d：%s"
                     % (cid, _st.get("slots", 0), _st.get("adopted", 0), _st.get("calls", 0),
                        "；".join("%s.%s %s→%s" % (t.get("key"), t.get("num"),
                                                   t.get("cur"), t.get("ans"))
                                  for t in (_st.get("trace") or []) if t.get("adopted"))))
        except Exception as exc:                               # noqa: BLE001
            _log(fp, "  ⚠️ %s 引脚名复读失败：%s: %s" % (cid, type(exc).__name__, exc))

    def _valread_step(cid, png):
        """在**已经落盘**的 `<cid>_Pred.json` 上补一层值重读；失败一律不影响导出。"""
        if not _valread_on:
            return
        dst_j = final_dir / ("%s_Pred.json" % cid)
        try:
            _ap = runs / "fusion" / "_assigned" / ("%s.json" % cid)
            if not _ap.exists():
                _ap = runs / "stage2" / "_assigned" / ("%s.json" % cid)
            _st = _vrmod.apply_file(png, _ap, dst_j, dst_j)
            if not _st:
                return
            valread_rows.append({"case": cid,
                                 **{k: v for k, v in _st.items() if k != "trace"}})
            if _st.get("adopted"):
                _log(fp, "  %-28s 值重读 槽 %d 采用 %d 调用 %d：%s"
                     % (cid, _st.get("slots", 0), _st.get("adopted", 0), _st.get("calls", 0),
                        "；".join("%s %s→%s" % (t.get("key"), t.get("cur"), t.get("ans"))
                                  for t in (_st.get("trace") or []) if t.get("adopted"))))
        except Exception as exc:                               # noqa: BLE001
            _log(fp, "  ⚠️ %s 值重读失败：%s: %s" % (cid, type(exc).__name__, exc))

    _n_seen = 0
    _fail_probe = 0
    _reread_skipped = []
    _broke = False
    try:
        _break_after = max(1, int(os.environ.get("VLM_BREAK_AFTER", "2")))
    except Exception:                                          # noqa: BLE001
        _break_after = 2
    for cid, _p in items:
        # ★ 熔断：`vlm_ok` 只在开跑前探过一次。复读通道每槽一次调用、失败**不抛异常**
        #   （`ask()` 返回 `"ERR:..."`），所以端点若在跑批途中死掉（LM Studio 崩、显存爆），
        #   后面每张图都会白等重试退避（实测 ≈15 s/槽）而整批分数一分不涨。每 20 张重探
        #   一次，探不通就整批关掉两个复读通道。
        _n_seen += 1
        if (_valread_on or _pinname_on) and _n_seen % 5 == 0:
            try:
                from vlm import client as _vc2                     # noqa: PLC0415
                # ★ 2026-10-13（§5.225/§5.227）：**单次探测失败不再熔断**，要连续
                #   `VLM_BREAK_AFTER` 次（默认 2）探不通才关。理由：§5.225 实测端点被
                #   别的任务占住时，第一次 `probe(timeout=30)` 会假失败，而熔断一旦触发
                #   就是**整批**失去值/引脚名两个通道（那台 220 张的机器上余下 216 张全没了，
                #   同参数同案子因此掉了 6.75 分）。代价：真死掉时多等一次探测（≤30 s）。
                if not _vc2.probe(timeout=30.0):
                    _fail_probe += 1
                    _log(fp, "⚠️ VLM 复读通道探测失败（第 %d 次，阈值 %d）"
                             % (_fail_probe, _break_after))
                    if _fail_probe >= _break_after:
                        _valread_on = _pinname_on = False
                        _broke = True
                        _log(fp, "⚠️ VLM 端点中途失效 ⇒ 熔断：关闭值/引脚名复读通道（余下 %d 张）"
                             % (len(items) - _n_seen + 1))
                else:
                    if _fail_probe:
                        _log(fp, "✅ VLM 复读通道探测恢复正常（连续失败清零，此前 %d 次）"
                             % _fail_probe)
                    _fail_probe = 0
            except Exception as _e:                                # noqa: BLE001
                _log(fp, "⚠️ 复读通道熔断探测本身失败：%s: %s" % (type(_e).__name__, _e))
        # 导出优先序：融合层 > VLM 补强 > 阶段三 > 阶段二。
        # 每一层都是「上一步的产物 + 一次受约束的改写」，所以取到哪一层都是完整的
        # 文档；取不到就往下退，保证 220 张里一张都不会因为某一层失败而丢产物。
        src = None
        for _d in (fuse_dir, s3v_dir, s3_dir, s2_dir):
            _cand = _d / ("%s.json" % cid)
            if _cand.exists():
                src = _cand
                break
        if src is None:
            # ★ 加固：以前这里只记一行日志就 `continue` ⇒ 该 case 在
            #   `final_results/` 里**一份文件都没有**。若评测端把「缺文件」当致命
            #   错误，就是整批 0 分。现在写一份空壳（三个键齐全，`metrics.py` 直接
            #   下标也解析得动），把「整批 0 分」压到「这一张 0 分」。
            ok = _write_shell(final_dir, cid, note="阶段一/二没有产物，空壳兜底")
            n_shell += 1 if ok else 0
            _log(fp, "  %-28s 没有产物（阶段一/二失败）→ %s"
                 % (cid, "已写空壳" if ok else "⚠️ 空壳也写失败"))
            continue
        # §5.82：引脚名「框选重读」通道（`FUSION_PINZOOM`，**默认 0 = 关**；A/B 实测
        #   80.98 → 80.01（−0.97），4 次采纳全坏 ⇒ 只留库，见 `vlm/pinzoom.py` 顶注）。它在
        #   **导出前**对最终文档再补一层受约束的名字改写（只碰名字、不动点/号/边）；失败
        #   一律不影响导出（退化成直接拷贝）。
        pz_note = ""
        dst = final_dir / ("%s_Pred.json" % cid)
        if _pinzoom_on:
            try:
                from vlm import pinzoom                            # noqa: PLC0415
                _st = pinzoom.apply_file(_p, src, dst)
                if _st is not None:
                    pinzoom_rows.append({"case": cid, **{k: v for k, v in _st.items()}})
                    pz_note = "（名字重读：槽 %d 采用 %d 调用 %d）" % (
                        _st.get("slots", 0), _st.get("adopted", 0), _st.get("calls", 0))
                    exported.add(cid)
                    n_export += 1
                    _log(fp, "  %-28s 引脚名重读 %s" % (cid, pz_note))
                    _valread_step(cid, _p)
                    _pinname_step(cid, _p)
                    _compdedup_step(cid)
                    _pwrclip_step(cid)
                    _edgecal_step(cid)
                    _pinkey_step(cid)
                    _symshift_step(cid)
                    _gndfill_step(cid, _p)
                    _anchor_step(cid)
                    _desigfill_step(cid, _p)
                    _holetype_step(cid)
                    _linefix_step(cid)
                    _degeom_step(cid)
                    _linemerge_step(cid)
                    _linejoin_step(cid)
                    _prune_step(cid)
                    _pinfill_step(cid)
                    continue
            except Exception as exc:                        # noqa: BLE001
                _log(fp, "  ⚠️ %s 引脚名重读失败：%s: %s" % (cid, type(exc).__name__, exc))
        shutil.copyfile(src, dst)
        exported.add(cid)
        n_export += 1
        _valread_step(cid, _p)
        _pinname_step(cid, _p)
        _compdedup_step(cid)
        _pwrclip_step(cid)
        _edgecal_step(cid)
        _pinkey_step(cid)
        _symshift_step(cid)
        _gndfill_step(cid, _p)
        _anchor_step(cid)
        _desigfill_step(cid, _p)
        _holetype_step(cid)
        _linefix_step(cid)
        _degeom_step(cid)
        _linemerge_step(cid)
        _linejoin_step(cid)
        _prune_step(cid)
        _pinfill_step(cid)
        if _broke and not (_valread_on or _pinname_on):
            _reread_skipped.append((cid, _p))
    _log(fp, "导出 %d 份 <case>_Pred.json%s"
         % (n_export, ("（另有 %d 份空壳兜底）" % n_shell) if n_shell else ""))

    # ------------------------------------------------- 复读第二遍（熔断后补救）
    # ★ 2026-10-13（§5.228）：熔断是**整批**的，而值/引脚名复读的价值是实打实的
    #   —— 220 张那一批里端点崩在融合层之后，211 张再没跑过复读，10GT 从 84.17 掉到
    #   76.89（−7.28）。端点崩掉往往只是一段负载（LM Studio 换模型/回收显存），
    #   等一会儿就活。所以跑完主循环后：**冷却 + 重探**，活了就把没跑过复读的案子补跑一遍。
    #   `_valread_step`/`_pinname_step` 都是从 `runs/**/_assigned` 重算再覆盖 dst，
    #   重复执行是幂等的，不会把两层改写叠加。
    if _reread_skipped and _broke and not (_valread_on or _pinname_on):
        try:
            _cool = float(os.environ.get("VLM_REREAD_COOLDOWN", "25"))
        except Exception:                                      # noqa: BLE001
            _cool = 25.0
        _log(fp, "⚠️ 有 %d 张没跑过值/引脚名复读（熔断）；冷却 %.0fs 后重探端点，"
                 "活了就补跑" % (len(_reread_skipped), _cool))
        _revived = False
        try:
            time.sleep(max(0.0, _cool))
            from vlm import client as _vc3                         # noqa: PLC0415
            _revived = bool(_vc3.probe(timeout=60.0))
        except Exception as _e:                                    # noqa: BLE001
            _log(fp, "⚠️ 复读第二遍探测失败：%s: %s" % (type(_e).__name__, _e))
        if _revived:
            _valread_on = _pinname_on = True
            _log(fp, "✅ 端点已恢复 ⇒ 第二遍补跑 %d 张的值/引脚名复读" % len(_reread_skipped))
            _n2 = 0
            for cid2, _p2 in _reread_skipped:
                _valread_step(cid2, _p2)
                _pinname_step(cid2, _p2)
                _n2 += 1
            _log(fp, "第二遍补跑完成：%d 张" % _n2)
        else:
            _log(fp, "⚠️ 端点仍未恢复 ⇒ 放弃第二遍（%d 张保持无复读）" % len(_reread_skipped))

    # ---------------------------------------------------------- 决策日志
    cases_log = []
    for cid, _p in items:
        e = {"case": cid, "exported": cid in exported}
        r1c = stage1_report.get(cid)
        if r1c is not None:
            e["stage1"] = {"ok": r1c["ok"], "timing": r1c.get("timing"),
                           "stats": r1c.get("stats")}
            if not r1c["ok"]:
                e["stage1"]["error"] = r1c.get("error")
        cases_log.append(e)
    s3_by = {r["case"]: r for r in s3_rows}
    for e in cases_log:
        r = s3_by.get(e["case"])
        if r is not None:
            e["stage3"] = {"n_slots": r["n_slots_total"], "n_named": r["n_named"],
                           "arb": r.get("arb"), "t": r["t"]}
    wall = round(time.time() - t_start, 1)
    # ★ 必须在 `fp.close()` **之前**写：第一版把这几行放在 close 之后，开通道时
    #   直接 `ValueError: I/O operation on closed file` ⇒ run.py 退出码 3（2026-10 A/B 实测）。
    if pinzoom_rows:
        _log(fp, "引脚名重读合计：槽 %d 采用 %d 调用 %d（%d 张图）"
             % (sum(r.get("slots", 0) for r in pinzoom_rows),
                sum(r.get("adopted", 0) for r in pinzoom_rows),
                sum(r.get("calls", 0) for r in pinzoom_rows), len(pinzoom_rows)))
    if valread_rows:
        _log(fp, "值重读合计：槽 %d 采用 %d 调用 %d（%d 张图）"
             % (sum(r.get("slots", 0) for r in valread_rows),
                sum(r.get("adopted", 0) for r in valread_rows),
                sum(r.get("calls", 0) for r in valread_rows), len(valread_rows)))
    if pinname_rows:
        _log(fp, "引脚名复读合计：槽 %d 采用 %d 调用 %d（%d 张图）"
             % (sum(r.get("slots", 0) for r in pinname_rows),
                sum(r.get("adopted", 0) for r in pinname_rows),
                sum(r.get("calls", 0) for r in pinname_rows), len(pinname_rows)))
    if compdedup_rows:
        _log(fp, "元件消重合计：删除 %d 个同物元件（%d 张图）"
             % (sum(r.get("dropped", 0) for r in compdedup_rows), len(compdedup_rows)))
    if pwrclip_rows:
        _log(fp, "网线吸附合计：吸 %d 个端点（%d 张图）"
             % (sum(r.get("moved", 0) for r in pwrclip_rows), len(pwrclip_rows)))
    if edgecal_rows:
        _log(fp, "端点内收合计：%d 条边（%d 张图）"
             % (sum(r.get("moved", 0) for r in edgecal_rows), len(edgecal_rows)))
    # ---------------- 尺度归一化的反向一步：把坐标乘回原图尺寸（`EDA_CANON_SCALE`）
    # 必须放在**所有**改写之后（复读第二遍也会重写 `<cid>_Pred.json`），且只碰坐标字段。
    if _canon_k and any(abs(_v - 1.0) > 1e-9 for _v in _canon_k.values()):
        _nsc = 0
        for _cid, _k in _canon_k.items():
            if abs(_k - 1.0) <= 1e-9:
                continue
            _f = final_dir / ("%s_Pred.json" % _cid)
            if not _f.exists():
                continue
            try:
                _doc = json.loads(_f.read_text(encoding="utf-8"))
                _scale_doc_xy(_doc, 1.0 / _k)
                _atomic_write_text(_f, json.dumps(_doc, ensure_ascii=False))
                _nsc += 1
            except Exception as _e:                                   # noqa: BLE001
                _log(fp, "  %-28s 坐标还原失败（%s）⇒ 该图保持归一化坐标" % (_cid, _e))
        _log(fp, "尺度还原：%d 份 <case>_Pred.json 的坐标乘回原尺寸" % _nsc)
    # `STAGE2_COMP_BOX`（单进程档的漏斗；lane 档由 `_publish_one` 兜住，两处都幂等）
    if compbox.mode() != compbox.OFF:
        _log(fp, "框模板（%s=%s）：%d 个占位框换成字形模板"
             % (compbox.ENV, compbox.mode(), compbox.apply_dir(final_dir)))
    _n_phantom = _drop_phantom_boxes(final_dir, fp)
    if _n_phantom:
        _log(fp, "零脚 box 清理：%d 个无引脚 IC 框" % _n_phantom)
    _n_strip = _strip_internal_keys(final_dir, fp)
    if _n_strip:
        _log(fp, "内部字段清理：%d 份 <case>_Pred.json 去掉 __pin_index/_pin_stubs" % _n_strip)
    _log(fp, "全部完成，墙钟 %.1fs（%.2f s/张）" % (wall, wall / max(1, len(items))))
    fp.close()

    _write_decision(out_dir,
                    {"n_cases": len(items), "n_stage1_ok": len(ok1),
                     "n_stage2_ok": len(s2_rows), "n_stage3_ok": len(s3_rows),
                     "n_exported": n_export, "n_shell": n_shell, "wall_s": wall,
                     "pinzoom": pinzoom_rows, "valread": valread_rows,
                     "pinname": pinname_rows, "compdedup": compdedup_rows,
                     "pwrclip": pwrclip_rows, "edgecal": edgecal_rows},
                    cases_log, a, t_start)
    print("产物 -> %s" % final_dir)
    return 0


def _atomic_write_text(path: Path, text: str) -> None:
    """先写同名 `.tmp` 再 `os.replace`（原子）。★ 加固：被硬杀在写一半时，
    留下的是一个**半截 JSON**，比没有更糟（评测端解析失败可能连累整批判定）。
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


# ★ 空壳产物：`schematic_parser/metrics.py` 对 `components`/`pins`/`nets` 是**直接下标**
#   （`:309 gc, pc = gt['components'], pred['components']`、`:455 flat(pred['pins'])`、
#   `:552 pred['nets'].items()`）⇒ 缺一个键就可能整批解析失败。空壳三个键都在，
#   得分是「这一张全 0」，而不是「整批 0」。用户 m42543 的灾难现场里这是底线。
_EMPTY_DOC = {"components": {}, "pins": {}, "nets": {}}


def _write_shell(final_dir: Path, cid: str, note: str = "") -> bool:
    """给一张**没有任何产物**的图写一份空壳 `<case>_Pred.json`（尽力而为）。"""
    try:
        doc = dict(_EMPTY_DOC)
        doc["case"] = cid
        if note:
            doc["note"] = note
        final_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(final_dir / ("%s_Pred.json" % cid),
                           json.dumps(doc, ensure_ascii=False))
        return True
    except Exception:                                               # noqa: BLE001
        return False


def _safety_net(a, exc: Exception) -> int:
    """**整进程兜底**（用户 m42543：「灾难性现场」）。`_run` 里任何未捕获异常都到这里。

    做的事（全部尽力而为，任何一步失败都不再抛）：
      1. 把异常与 traceback 打进 `final_results/run.log`；
      2. 对输入目录里**每一张图**保证有 `<case>_Pred.json`（已有的不动，缺的写空壳）；
      3. 写 `decision_log.json`（含 `fatal` 字段与 traceback）。
    `main()` 以前是**整个函数体外无 try**（`run.py:71`），模块级 import 一炸
    （`:224 from stage1.detect import DEFAULT_WEIGHTS`）就一份产物都没有、退出码 0。
    """
    try:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    except Exception:                                               # noqa: BLE001
        pass
    out_dir = Path(getattr(a, "output_dir", ".") or ".")
    final_dir = out_dir / "final_results"
    n_shell = 0
    try:
        final_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (final_dir / "run.log").open("a", encoding="utf-8") as fp:
                _log(fp, "!!! 致命异常 %s: %s" % (type(exc).__name__, exc))
                fp.write(traceback.format_exc()[-4000:] + "\n")
        except Exception:                                           # noqa: BLE001
            pass
        try:
            in_dir = Path(getattr(a, "input_dir", ".") or ".")
            for cid, _p in _dedupe_items(discover_inputs(in_dir))[0]:
                if not (final_dir / ("%s_Pred.json" % cid)).exists():
                    if _write_shell(final_dir, cid, note="run.py 致命异常，空壳兜底"):
                        n_shell += 1
        except Exception:                                           # noqa: BLE001
            pass
    except Exception:                                               # noqa: BLE001
        pass
    try:
        _write_decision(out_dir, {"fatal": "%s: %s" % (type(exc).__name__, exc),
                                  "n_shell": n_shell}, [], a, time.time(),
                        note="fatal")
    except Exception:                                               # noqa: BLE001
        pass
    return 3


def _write_decision(out_dir: Path, summary: dict, cases_log: list,
                    a, t_start: float, note: str = ""):
    """写 `/output/decision_log.json`。

    ⚠️ 官方手册（EDARace）里 `decision_log.json` 的字段定义我们手上没有，
    所以这里写成**自描述、可机读**的一份：顶层带 `schema` 与 `note`，把
    配置、权重、耗时、逐 case 统计与仲裁结果全放进去。评测端若只检查
    「文件存在且是合法 JSON」，这已经满足；若手册另有字段要求，改这一个函数即可。
    """
    doc = {
        "schema": "edarace-decision-log/1",
        "note": "字段定义以官方 EDARace 手册为准；本文件按自描述格式写出，"
                "含配置、权重解析、逐 case 统计与 LLM 仲裁记录。" + ((" " + note) if note else ""),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t_start, 1),
        "input_dir": str(a.input_dir),
        "output_dir": str(a.output_dir),
        "config": {
            "device": a.device, "imgsz": a.imgsz, "workers": a.workers,
            "tile": int(getattr(a, "tile", 0) or 0),
            "overlap": int(getattr(a, "overlap", 0) or 0),
            "case_workers": getattr(a, "case_workers", -1),
            "do_wires": not a.no_wires, "texts": a.texts,
            "llm_mode": a.mode, "llm_batch": a.llm_batch,
            "llm_max_slots": a.llm_max_slots,
            "vlm": a.vlm, "vlm_workers": a.vlm_workers, "vlm_margin": a.vlm_margin,
            "vlm_desig_fallback": a.vlm_desig_fallback,
            # ★ 生效档位的**逐项**明细（§5.51）：平台回传里只有这个文件，而
            #   `vlm_stats` 看不出「引脚通道关没关」（关掉时它同样报 grid_calls=0）；
            #   把 `_apply_lean` / `_apply_offline` 之后的实际值写下来，事后能复盘档位。
            "vlm_pin": a.vlm_pin, "vlm_pin_name": a.vlm_pin_name,
            "vlm_name": a.vlm_name, "vlm_value": a.vlm_value,
            "vlm_value_batch": a.vlm_value_batch,
            "fusion": a.fusion,
            "vlm_stats": getattr(a, "vlm_stats", None),
            # ★ 交付护栏的实际生效值 + 外部推理的真实用量（超预算 ⇒ `refused>0`）。
            #   只读**本进程已经 import 过**的 `vlm.client`：父进程（扇出）没跑过 VLM，
            #   这里如实报 `None`，不去凭空 import 一个「零用量」的模块骗自己。
            "platform_budgets": getattr(a, "platform_budgets", None),
            "vlm_client": (lambda m: m.stats() if m is not None else None)(
                sys.modules.get("vlm.client")),
        },
        "env": {k: os.environ.get(k, "") for k in
                ("LLM_MODEL", "LLM_BASE_URL", "NVIDIA_VISIBLE_DEVICES",
                 "STAGE2_BOTH_TOL", "STAGE2_DROP_UNKEYED", "STAGE2_NET_DROP_UNKEYED",
                 "STAGE2_PIN_DROP_UNKEYED",
                 "FUSION_REF_PRUNE", "FUSION_MODEL_SET",
                 "FUSION_MODEL_SET_MIN", "FUSION_MODEL_SET_MARGIN",
                 "FUSION_NAME_SNAP",
                 "FUSION_NAME_SNAP_TOL", "FUSION_NAME_SNAP_MINLEN",
                 "FUSION_VALUE_FIX",
                 "FUSION_PIN_LIBFIX", "FUSION_PIN_LIBFIX_RATIO",
                 "FUSION_PIN_LIBFIX_MARGIN",
                 "VLM_TIMEOUT", "VLM_BUDGET_S",
                 # ★ 比赛日档位：由交付入口 `submission/agent_system` 读进参数
                 #   （`docs/RESULTS.md` §5.51），这里只登记它，便于判题机回传里
                 #   直接看出那一轮跑的是哪个档。
                 "EDA_TIER_ARGS",
                 "EDA_WALL_BUDGET_S", "EDA_CASE_WORKERS", "EDA_PUBLISH_S")},
        "assets": assets.describe(),
        "summary": summary,
        "cases": cases_log,
    }
    _atomic_write_text(out_dir / "decision_log.json",
                       json.dumps(doc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

