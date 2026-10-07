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
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
                    help="stage1 进程池大小（CPU 密集；>1 会各自占显存）")
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
    ap.add_argument("-vlm-desig-fallback", "--vlm-desig-fallback",
                    dest="vlm_desig_fallback", default="on", choices=("on", "off"),
                    help="对『键还是兜底键 pNNN』的元件补读位号（不限类型）。"
                         "220 张里这类元件 2601 个，其中 92%% 在 10px 内就有文字框，"
                         "但 `assign.py` 的全局去重把那个框判给了别人；而 pNNN 键"
                         "永远不可能和 GT 对上，换掉只有赚没有赔。")
    ap.add_argument("-fusion", "--fusion", dest="fusion", default="on",
                    choices=("off", "on", "llm"),
                    help="融合层（21 节点架构的落地）：先验 + 解 + 合理性检查。"
                         "`on` = 只跑确定性节点（不调模型，220 张约 10 分钟）；"
                         "`llm` = 再叠加判官/审计/重扫（每张多约 50 秒 LM Studio）；"
                         "`off` = 不做融合，直接导出阶段三/二产物。"
                         "220 张实测：融合 `on` = 28.4200、不融合（VLM 补强后的阶段二）"
                         "= 28.4133 ⇒ 融合**不降分**（+0.007，噪声级），而它修掉了"
                         "`Name` 顶位号这类真 bug。")
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
    a = ap.parse_args(argv)

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
    _log(fp, "权重 " + json.dumps(assets.describe(), ensure_ascii=False))

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
    t0 = time.time()
    for i, cid in enumerate([c.case_id for c in cases if c.case_id in ok1], 1):
        try:
            r = s2_process(cid, s1_dir, s2_dir, engine, texts=a.texts)
            s2_rows.append(r)
            _log(fp, "  [%d] %-28s 元件 %-4d 位号 %-4d %.1fs"
                 % (i, cid, r["n_components"], r["n_designator"], r["t_total"]))
        except Exception as e:                                    # noqa: BLE001
            _log(fp, "  [%d] %-28s 阶段二失败 %s: %s" % (i, cid, type(e).__name__, e))
            fp.write(traceback.format_exc()[-800:] + "\n")
    _log(fp, "阶段二完成 %d 张，用时 %.1fs" % (len(s2_rows), time.time() - t0))

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
                t0 = time.time()
                on = vlm_client.probe()
                _log(fp, "VLM 端点探测（%s，%.1fs）：%s"
                     % (vlm_client.BASE, time.time() - t0, "可用" if on else "不支持图片，跳过"))
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
            else:
                # 2026-07：这里原来是 `elif a.vlm == "on":`，挂在 `if a.vlm != "off"` 的
                # else 分支里 —— `a.vlm == "on"` 必然满足 `!= "off"`，所以那条警告
                # **永远打不出来**；真正需要告警的「auto 探测失败 ⇒ 静默不补强」反而无声。
                _log(fp, "警告：-vlm %s 但补强未执行（端点不可用），导出的产物**未经 VLM 补强**"
                     % a.vlm)
        except Exception as exc:                    # noqa: BLE001
            import traceback
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
        os.environ["FUSION_CROPFIX"] = "1" if a.fusion == "llm" else "0"
        if a.fusion == "off":
            os.environ["FUSION_LLM"] = "0"
        elif a.fusion == "llm":
            os.environ["FUSION_LLM"] = "1"
        _do_llm = os.environ.get("FUSION_LLM", "1").strip().lower() \
            not in ("0", "off", "no", "false", "none", "")
        try:
            from fusion import run as fusion_run
            want = [r["case"] for r in s2_rows]
            _log(fp, "=== 融合层：先验 + 解 + 合理性检查（%d 张，LLM 轮 %s，cropfix %s） ==="
                 % (len(want), "开" if _do_llm else "关",
                    "开" if a.fusion == "llm" else "关"))
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
        except Exception as exc:                        # noqa: BLE001
            _log(fp, "⚠️ 融合层整段失败：%s: %s —— 继续用未融合的产物导出"
                 % (type(exc).__name__, exc))
            fp.write(traceback.format_exc()[-800:] + "\n")
    a.fusion_stats = {"n": len(fusion_rows), "mode": a.fusion}

    # ---------------------------------------------------------- 导出
    _log(fp, "=== 导出 final_results ===")
    n_export = 0
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
            _log(fp, "  %-28s 没有产物（阶段一/二失败）" % cid)
            continue
        shutil.copyfile(src, final_dir / ("%s_Pred.json" % cid))
        exported.add(cid)
        n_export += 1
    _log(fp, "导出 %d 份 <case>_Pred.json" % n_export)

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
                     "n_exported": n_export, "wall_s": wall},
                    cases_log, a, t_start)
    print("产物 -> %s" % final_dir)
    return 0


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
            "do_wires": not a.no_wires, "texts": a.texts,
            "llm_mode": a.mode, "llm_batch": a.llm_batch,
            "llm_max_slots": a.llm_max_slots,
            "vlm": a.vlm, "vlm_workers": a.vlm_workers, "vlm_margin": a.vlm_margin,
            "vlm_desig_fallback": a.vlm_desig_fallback,
            "vlm_stats": getattr(a, "vlm_stats", None),
        },
        "env": {k: os.environ.get(k, "") for k in
                ("LLM_MODEL", "LLM_BASE_URL", "NVIDIA_VISIBLE_DEVICES",
                 "STAGE2_BOTH_TOL", "STAGE2_DROP_UNKEYED", "STAGE2_NET_DROP_UNKEYED")},
        "assets": assets.describe(),
        "summary": summary,
        "cases": cases_log,
    }
    (out_dir / "decision_log.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
