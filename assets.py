# -*- coding: utf-8 -*-
"""模型权重的**可移植**查找 —— 提交到评测容器时不能有写死的绝对路径。

原来的写法是写死 `Y:\\workspace\\EDA\\...`，在本机跑得通，一进容器就
`FileNotFoundError`（而且是在 stage1 的第一行、整批 case 全灭）。这里统一成：

    环境变量覆盖  ->  包内相对路径（按顺序试）  ->  报错并列出所有试过的位置

两个权重：

* **YOLO 元件检测** `pcb43_v2_from_best`：随仓库一起提交，在 `yolo_part/runs/` 下。
* **CRNN 文字识别** `ocr_mix/runs/best.pt`（2026-10-11 起；此前是 `ocr_real/runs/best.pt`，
  e51）：训练产物原本落在 `work\\`（**不在包内**），已复制一份到 `weights/textrec_best.pt`
  随仓库提交（e57，md5 `7559096C9979`）。
* **官方元器件库 `cpntLibrary`**（1,755,587 B / 337 键的 JSON，`stage2/libpins.py` 的引脚
  先验要靠它）：原本只放在本机的 `WORKSPACE\work\data\`（`work\` **不进仓库、不进镜像**）
  ⇒ 容器里按旧默认路径去找必然不存在，`STAGE2_LIB_PINS` 虽然默认开，也只会印一行
  「库不可用」然后**静默跳过** —— 白丢实测 +1.60（官方 9 张，交付链）/ +2.61（官方 10GT）。
  已复制一份到 `data/cpntLibrary.json` 随仓库提交，并把它并入本模块的候选链。
* **官方 box/IC 库 `BoxICLibrary_200cases.json`**（1,826,150 B / 360 键，用户 m42932
  「以新的这个为参考」）：同一套 `{esym_path, pins, same_hash_components}` schema，
  但**覆盖的是 box/IC 那一档**，而官方 `pin_f1` 只考 box 元件。同口径实测（`_libcov.py`，
  只统计能命中库的元件）：IC 名命中率 官方 9 张 **85.7% vs 71.4%**、200 训练案例
  **61.6% vs 30.9%**；引脚名覆盖 **99.2%/69.7% vs 99.0%/64.9%**。⇒ 与 `cpntLibrary`
  一起进包，由 `stage2/libpins.py` 合并使用（BoxIC 优先、cpnt 补缺键）。

Docker 里只要把整个 `eda-pcb` 拷进 `/submission`，这些外部数据（两个权重 + 两份元件库）
都在相对位置能找到。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

# `eda-pcb/` 包根
REPO = Path(__file__).resolve().parent
# `EDA/` 工作区根（`eda-pcb` 的上一级）；容器里通常不存在，只作为本机兜底
WORKSPACE = REPO.parent

YOLO_ENV = "YOLO_WEIGHTS"
CRNN_ENV = "STAGE2_CKPT"
CPNT_ENV = "STAGE2_CPNT_LIB"
BOXIC_ENV = "STAGE2_BOXIC_LIB"


def _candidates(kind: str) -> List[Path]:
    if kind == "yolo":
        return [
            REPO / "weights" / "pcb43_v2_from_best.pt",
            REPO / "yolo_part" / "runs" / "detect" / "pcb43_v2_from_best" / "weights" / "best.pt",
        ]
    if kind == "cpntlib":
        # ① 包内（随仓库提交 ⇒ 进镜像的唯一保证）；②③④ 只在本机开发时兜底。
        return [
            REPO / "data" / "cpntLibrary.json",
            WORKSPACE / "work" / "data" / "cpntLibrary.json",
            WORKSPACE / "赛题资料" / "cpntLibrary",
            REPO / "yolo_part" / "Dataset(waitforpreprocess)" / "cpntLibrary",
        ]
    if kind == "boxic":
        # 官方 box/IC 库（用户 m42932「以新的这个为参考」）。同 cpntlib：包内优先。
        return [
            REPO / "data" / "BoxICLibrary_200cases.json",
            WORKSPACE / "work" / "data" / "BoxICLibrary_200cases.json",
            WORKSPACE / "赛题资料" / "BoxICLibrary_200cases.json",
            REPO / "yolo_part" / "Dataset(waitforpreprocess)" / "BoxICLibrary_200cases.json",
        ]
    return [
        REPO / "weights" / "textrec_best.pt",
        WORKSPACE / "work" / "ocr_mix" / "runs" / "best.pt",
        WORKSPACE / "work" / "ocr_real" / "runs" / "best.pt",
    ]


def resolve(kind: str, env: str, extra: Iterable[Path] = (), must_exist: bool = True) -> str:
    """返回权重路径。`kind` ∈ {``"yolo"``, ``"textrec"``}。"""
    tried: List[str] = []
    v = os.environ.get(env)
    if v:
        p = Path(v)
        if p.exists():
            return str(p)
        tried.append("%s=%s（不存在）" % (env, v))
    for p in list(extra) + _candidates(kind):
        tried.append(str(p))
        if p.exists():
            return str(p)
    if not must_exist:
        # 允许缺席：`stage1.detect` 会在真正用到时才失败，`stage2` 则允许无 OCR 跑通
        return str(_candidates(kind)[0])
    raise FileNotFoundError(
        "%s 权重找不到。试过：\n  " % kind + "\n  ".join(tried)
        + "\n可用环境变量 %s 指定。" % env)


def yolo_weights(must_exist: bool = True) -> str:
    return resolve("yolo", YOLO_ENV, must_exist=must_exist)


def textrec_ckpt(must_exist: bool = True) -> str:
    return resolve("textrec", CRNN_ENV, must_exist=must_exist)


def cpnt_library(must_exist: bool = False) -> str:
    """官方元器件库（`cpntLibrary`，1.76 MB JSON）的路径。

    `must_exist=False`（默认）是**故意的**：库只是 `stage2/libpins.py` 的一条先验通道，
    缺席时必须能让管线继续跑（那边会打印一行并静默跳过），绝不能像权重那样抛异常
    把整批 case 打掉。默认返回候选链里的**第一个**位置（即使它不存在），
    这样调用方总能拿到一个「本该在哪」的路径用于报错/日志。
    """
    return resolve("cpntlib", CPNT_ENV, must_exist=must_exist)


def boxic_library(must_exist: bool = False) -> str:
    """官方 box/IC 元件库（`BoxICLibrary_200cases.json`，1.83 MB / 360 键）的路径。

    用户 m42932「以新的这个为参考」。与 `cpnt_library()` 同样是**允许缺席**的：
    `stage2/libpins.py` 会把两份库合并使用（BoxIC 优先、cpnt 补缺键），
    少一份只是覆盖变小，绝不能让整批 case 挂掉。
    """
    return resolve("boxic", BOXIC_ENV, must_exist=must_exist)


def libraries() -> List[str]:
    """`stage2/libpins.py` 要合并的库文件列表（按优先级，**前面的覆盖后面的**）。

    顺序是 BoxIC 在前：官方 `pin_f1` 只考 box 元件，而 BoxIC 就是 box/IC 那一档的库
    （IC 名命中率 85.7% vs 71.4%（官方 9 张）/ 61.6% vs 30.9%（200 案例））。
    任一份不存在就跳过（`must_exist=False`），两份都没有时返回空表 ⇒ 调用方静默跳过本通道。
    """
    out: List[str] = []
    for fn in (boxic_library, cpnt_library):
        try:
            p = fn(must_exist=False)
        except Exception:                                        # noqa: BLE001
            continue
        if p and Path(p).exists() and p not in out:
            out.append(p)
    return out


def describe() -> dict:
    out = {}
    for kind, env, fn in (("yolo", YOLO_ENV, yolo_weights),
                          ("textrec", CRNN_ENV, textrec_ckpt),
                          ("cpntlib", CPNT_ENV, lambda: cpnt_library(must_exist=True)),
                          ("boxic", BOXIC_ENV, lambda: boxic_library(must_exist=True))):
        try:
            p = fn()
            out[kind] = {"path": p, "exists": Path(p).exists(),
                         "MB": round(Path(p).stat().st_size / 1e6, 1) if Path(p).exists() else 0}
        except Exception as e:                                   # noqa: BLE001
            out[kind] = {"error": str(e)}
    return out


if __name__ == "__main__":
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(describe(), ensure_ascii=False, indent=1))
