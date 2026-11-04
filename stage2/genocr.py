# -*- coding: utf-8 -*-
"""通用 OCR（PP-OCR / `rapidocr_onnxruntime`）第三识别通道：**只跑识别头**。

动机（`work/_genocr_boxes.py`，2026-11 实测）：我方 CRNN 与通用 OCR 的误差不相关 ——
池 82 件「全页没读到」的框上，我方 exact 20/82、CER 0.3586，PP-OCR 52/82、CER 0.1514；
13 页 21 条文本错配里通用读法读对 12 条（`case1 U2 SX1308`、`case11 U1 TP4333`、
`case13 C3 22uf` …）。**全页检测要 ~30 s，太慢，不用**：这里只复用 stage1 已有文字框，
裁片 = 框外扩 `PAD` 像素、`Image.LANCZOS` 放大 `UP` 倍，只跑 rec 头 ≈10 ms/框。

缺包（`ImportError`）时 `available()` 返回 False，调用方必须原样放行 —— 管线照常跑。
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]

#: 与 `work/_genocr_boxes.py` 逐位相同的裁片参数（那批实测就是在这组值上取到的）。
PAD = 3
UP = 3
_MIN_SIDE = 4

_ENV: Optional[str] = None
_ENG = None
_TRIED = False


def _engine():
    """懒加载 `RapidOCR` 单例；导入/初始化失败只记一次，之后一路返回 None。"""
    global _ENG, _TRIED, _ENV
    if _TRIED:
        return _ENG
    _TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ENG = RapidOCR()
    except Exception as exc:                                   # noqa: BLE001
        _ENV = repr(exc)
        _ENG = None
    return _ENG


def available() -> bool:
    """缺包/初始化失败 ⇒ False（管线必须照常跑，不能崩）。"""
    return _engine() is not None


def note() -> str:
    """诊断串：`available()` 失败时的原因。"""
    return _ENV or ""


def _rec_one(eng, gray: np.ndarray, box: Box) -> Tuple[str, float]:
    hi, wi = gray.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD)
    x1, y1 = min(wi, x1 + PAD), min(hi, y1 + PAD)
    if x1 - x0 < _MIN_SIDE or y1 - y0 < _MIN_SIDE:
        return "", 0.0
    try:
        from PIL import Image
        sub = gray[y0:y1, x0:x1]
        if sub.ndim == 3:                                  # BGR -> RGB（PIL 的通道序）
            sub = sub[:, :, ::-1]
        img = Image.fromarray(np.ascontiguousarray(sub))
        img = img.convert("RGB").resize((img.width * UP, img.height * UP), Image.LANCZOS)
        got, _ = eng(np.asarray(img), use_det=False, use_cls=False, use_rec=True)
    except Exception:                                          # noqa: BLE001
        return "", 0.0
    if not got:
        return "", 0.0
    txt = (got[0][0] or "").strip()
    try:
        conf = float(got[0][1])
    except (TypeError, ValueError, IndexError):
        conf = 0.0
    return txt, conf


def read_many(gray: np.ndarray,
              items: Sequence[Tuple[Box, object]]) -> List[Tuple[str, float]]:
    """与 `textrec.infer.Recognizer.read_many` 同形：`[(box, role)] -> [(text, conf)]`。

    `gray` 是整页 8bit 灰度（或 BGR，`_rec_one` 会换通道序）。`role` 这里**不参与**
    裁剪/识别，只为了与调用方保持同一张表；`fix_confusions` 由调用方按角色另调。
    """
    eng = _engine()
    if eng is None:
        return [("", 0.0) for _ in items]
    out: List[Tuple[str, float]] = []
    for box, _role in items:
        out.append(_rec_one(eng, gray, box))
    return out


# ---------------------------------------------------------------------------
#: 极小混淆归一化。**只作用于通用读法本身**（绝不改我方 CRNN 的读法）。
_CONF_DIGIT = str.maketrans({"I": "1", "l": "1", "O": "0", "o": "0", "S": "5", "s": "5"})
#: 值候选：整个串就是「数字（可带小数点）+ 至多一个尾部单位字母」。只有这种串里
#: 把 `I/l/O/S` 当数字才可能对；落在 `SOIC` / `AO3400` / `SX1308` 上是净损失。
_NUMTOK_RE = re.compile(r"^[0-9IlOoSs]+(?:\.[0-9IlOoSs]+)?[A-Za-z]?$")
#: 型号候选：开头是小写 `i`/`l`、后面紧跟大写字母或数字（`iN4007` -> `1N4007`）。
#: 真型号不会以小写字母开头，大写开头的 `L`/`I`/`O`/`S` 一律不碰。
_MODEL_LEAD_RE = re.compile(r"^[il](?=[A-Z0-9])")


def fix_confusions(text: str, role: str) -> str:
    """`I/l→1`、`O→0`、`S→5`，只对「纯数字+单位」与「型号」两类候选串。

    · `value`：`1O4` -> `104`、`l0k` -> `10k`（整串必须先是数字 token 形状）；
    · `partname`：`iN4007` -> `1N4007`（仅**开头小写** `i/l`）；
    · 其余角色 / 其余形状原样返回。
    """
    t = (text or "").strip()
    if not t or role not in ("value", "partname"):
        return t
    if role == "partname":
        if not _MODEL_LEAD_RE.match(t):
            return t
        return "1" + t[1:]
    if not _NUMTOK_RE.match(t) or not any(c.isdigit() for c in t):
        return t
    return t.translate(_CONF_DIGIT)
