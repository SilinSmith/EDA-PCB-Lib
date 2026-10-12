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
import json
import os
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
    ap.add_argument("-vlm", "--vlm", dest="vlm", default="auto",
                    choices=("auto", "on", "off"),
                    help="阶段二之后的 VLM 补强（读 IC 位号 + 引脚号/名）。"
                         "auto = 先探一次端点收不收图；需要视觉模型。实测 +0.565。")
    ap.add_argument("-vlm-workers", "--vlm-workers", dest="vlm_workers", type=int,
                    default=4, help="VLM 并发数。默认从 6 降到 4：LM Studio 与管线"
                                    "同卡时，并发请求会让 LLM 自己的 KV cache 成倍涨。"
                                    "实测 4~6 路都可用，4 更稳。")
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
                         "**默认开**：官方 10GT 9 张交付口径 58.6304 → **61.5112（+2.88）**"
                         "（pin F1 0.3920→0.4858、tp 37→45）；全量 220 按 split 拆开是"
                         " 10GT **+2.59** / 10_GTcase **+0.76** / 200_train_cases **−0.25**"
                         "（负号只在自动命名那一档）。⚠ 这是**替换**引脚键那条路"
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
    if not getattr(a, "offline", False):
        return
    a.vlm = "off"
    # ★ 只关 LLM 轮，**不关融合层**：融合层里的确定性节点（先验 + 解）本身不发请求，
    #   实测值 +0.24（`zero` 档 70.04 vs `-fusion off` 的 69.80，官方 10 张）。
    #   `-fusion on`（默认）不覆盖 `FUSION_LLM` ⇒ 这里显式置 0 即可。
    for k, v in (("FUSION_LLM", "0"), ("FUSION_ITERATE", "0"),
                 ("STAGE1_WIRE_COLOUR", "dominant"), ("STAGE2_PINROT", "0")):
        if k not in os.environ:
            os.environ[k] = v
    print("[run] 离线档：-vlm off｜FUSION_LLM=%s｜FUSION_ITERATE=%s｜STAGE1_WIRE_COLOUR=%s"
          "｜STAGE2_PINROT=%s（零外部调用，保留融合层确定性节点）"
          % (os.environ.get("FUSION_LLM"), os.environ.get("FUSION_ITERATE"),
             os.environ.get("STAGE1_WIRE_COLOUR"), os.environ.get("STAGE2_PINROT")),
          flush=True)


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
    ("fusion", "-fusion"), ("llm_max_slots", "-llm-max-slots"),
    ("gpu_frac", "-gpu-frac"), ("gpu_headroom", "-gpu-headroom"),
)


def _visible_cuda_devices() -> int:
    try:
        import torch
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:                                               # noqa: BLE001
        return 0


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
    if ndev <= 1:
        return 1                       # 单卡/无卡：保持交付默认（逐 case 串行）
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
        json.loads(data.decode("utf-8"))
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
    child_vw = int(getattr(a, "vlm_workers", 4))
    if n_lanes >= 4:
        child_vw = max(1, child_vw // 2)
    child_s1 = int(getattr(a, "workers", 1))
    if n_lanes >= 4:
        child_s1 = 1                     # 每 lane 的 stage1 交给 lane 之间的并行
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
    if a.vlm == "auto":
        try:
            from vlm import client as _vc
            _t0 = time.time()
            vlm_ok = bool(_vc.probe())
            _log(fp, "VLM 端点预探测（%s，%.1fs）：%s"
                 % (_vc.BASE, time.time() - _t0,
                    "可用" if vlm_ok else "不支持图片，跳过补强"))
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
                   imgsz=a.imgsz, conf=0.25, device=a.device,
                   use_cv=True, do_wires=not a.no_wires)
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
                shutil.copyfile(s2_dir / ("%s.json" % cid),
                                final_dir / ("%s_Pred.json" % cid))
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
                     % (vlm_client.BASE, "可用" if on else "不支持图片，跳过"))
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
                          "desig_fallback": (a.vlm_desig_fallback == "on")},
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
                         "｜token 入 %d / 出 %d / 合计 %d%s"
                     % (_bs["calls"], _bs["errors"], _bd,
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
    for cid, _p in items:
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
        shutil.copyfile(src, final_dir / ("%s_Pred.json" % cid))
        exported.add(cid)
        n_export += 1
    _log(fp, "导出 %d 份 <case>_Pred.json%s"
         % (n_export, ("（另有 %d 份空壳兜底）" % n_shell) if n_shell else ""))

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
    _log(fp, "全部完成，墙钟 %.1fs（%.2f s/张）" % (wall, wall / max(1, len(items))))
    fp.close()

    _write_decision(out_dir,
                    {"n_cases": len(items), "n_stage1_ok": len(ok1),
                     "n_stage2_ok": len(s2_rows), "n_stage3_ok": len(s3_rows),
                     "n_exported": n_export, "n_shell": n_shell, "wall_s": wall},
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
                 "STAGE2_PIN_DROP_UNKEYED", "STAGE2_PIN_DROP_LAUNDERED",
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
