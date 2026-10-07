# -*- coding: utf-8 -*-
"""模型权重的**可移植**查找 —— 提交到评测容器时不能有写死的绝对路径。

原来的写法是写死 `Y:\\workspace\\EDA\\...`，在本机跑得通，一进容器就
`FileNotFoundError`（而且是在 stage1 的第一行、整批 case 全灭）。这里统一成：

    环境变量覆盖  ->  包内相对路径（按顺序试）  ->  报错并列出所有试过的位置

两个权重：

* **YOLO 元件检测** `pcb43_v2_from_best`：随仓库一起提交，在 `yolo_part/runs/` 下。
* **CRNN 文字识别** `ocr_real/runs/best.pt`：训练产物原本落在 `work\\`（**不在包内**），
  已复制一份到 `weights/textrec_best.pt` 随仓库提交。

Docker 里只要把整个 `eda-pcb` 拷进 `/submission`，两个权重都在相对位置能找到。
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


def _candidates(kind: str) -> List[Path]:
    if kind == "yolo":
        return [
            REPO / "weights" / "pcb43_v2_from_best.pt",
            REPO / "yolo_part" / "runs" / "detect" / "pcb43_v2_from_best" / "weights" / "best.pt",
        ]
    return [
        REPO / "weights" / "textrec_best.pt",
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


def describe() -> dict:
    out = {}
    for kind, env, fn in (("yolo", YOLO_ENV, yolo_weights),
                          ("textrec", CRNN_ENV, textrec_ckpt)):
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
