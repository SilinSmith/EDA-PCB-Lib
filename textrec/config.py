# -*- coding: utf-8 -*-
"""textrec 全局配置：路径、角色表、图像超参。

所有路径都用 Path 拼，脚本不依赖当前工作目录。
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDA = HERE.parent                      # Y:\workspace\EDA\eda-pcb
WORKSPACE = EDA.parent                 # Y:\workspace\EDA

# ---------------- 合成语料（训练集的主要来源） ----------------
SYNTH_ROOT = WORKSPACE / "合成原理图数据集"
SYNTH_IMAGES = SYNTH_ROOT / "images"
SYNTH_TEXTS = SYNTH_ROOT / "texts.jsonl"

# ---------------- 真实语料（唯一可用于「真实泛化」验证的来源） ----------------
YOLO_PART = EDA / "yolo_part"


def _find_dataset300() -> "Path | None":
    """Dataset300 的目录名带中文后缀，用前缀匹配，避免写死。"""
    if not YOLO_PART.is_dir():
        return None
    for p in sorted(YOLO_PART.iterdir()):
        if p.is_dir() and p.name.startswith("Dataset300"):
            return p
    return None


DATASET300 = _find_dataset300()

# ---------------- 产物 ----------------
DATA_DIR = HERE / "data"
RUNS_DIR = HERE / "runs"
MANIFEST = DATA_DIR / "manifest.jsonl"
CHARSET_PATH = DATA_DIR / "charset.json"

# ---------------- 文本角色 ----------------
# 顺序必须与 synth/emit.py 的 TEXT_ROLES 一致（它决定 labels_text 的类别号）。
ROLES = ("designator", "value", "pin_name", "pin_number", "net_label", "note")
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
N_ROLES = len(ROLES)

# ---------------- 图像超参 ----------------
IMG_H = 32            # 所有裁剪统一缩放到这个高度
MAX_W = 384           # 宽度上限（超过就压缩）
MIN_W = 8
# CNN 对宽度的总下采样倍数。必须与 model.py 的池化配置一致。
# 它决定了 CTC 的时间步数：T = W // WIDTH_REDUCTION。
# 语料里最长的文本是 16 字符（value，如 "crystal_2pin"），其宽高比约 7.2，
# 缩到高 32 后宽约 230，T≈28，够 CTC 编码 16 个字符。
WIDTH_REDUCTION = 8

# 文本长度上限。真实语料里有 58 字符的整段说明文字（note/value），
# 它们既不是位号也不是参数，缩到固定高度后宽高比会严重失真，直接丢掉。
MAX_TEXT_LEN = 24

# 裁剪时向外留的边距（按文字高度的比例）。
PAD_X_FRAC = 0.12
PAD_Y_FRAC = 0.25

# ---------------- 墨迹垂直归一（★ 2026-10-07）----------------
# 背景：`crop_gray` 把裁剪**缩放到高 IMG_H=32**，所以模型看到的字形高度＝
#   「框内字形高度 / 框高」× 32。而框是从外面来的：
#     · 标注框（`work/ocr_real/manifest.jsonl`，训练侧）是松的；
#     · stage1 的文字框是**按墨迹阈值取的紧框**（中位 IoU 0.36），字形会被切掉。
#   两边「框高」差一大截 ⇒ 同一个模型在两种框上表现天差地别。实测（同一份
#   796 条真实 val，只改 `stage2/text.py` 的 `pad_box`）：
#       线上默认 pad(水平2.0/0.30, 垂直2.0/0.30)   exact 0.197  位号 0.239
#       外扩全 0（= 训练侧口径）                   exact 0.661  位号 0.823
#   但把外扩全 0 用到**管线的紧框**上，220 张 10GT 端到端从 48.60 掉到 24.54
#   —— 紧框被切字，改不动。也就是说：**光调外扩参数解决不了，因为两边需要的
#   外扩量本来就不同**。
#
# 解法：不再依赖外面的框有多紧，**在裁剪内部把垂直范围对齐到墨迹**，让字形高度
# 恒定居裁剪高度的 `1/(1+2*PAD_Y_FRAC) ≈ 2/3`。这样训练框（松）与 stage1 紧框
# 会得到**同一种**裁剪，模型不再需要对外扩量鲁棒。
INK_YNORM = os.environ.get("TEXTREC_INK_YNORM", "0").strip() not in (
    "", "0", "false", "off", "none", "no")
INK_THRESH = int(os.environ.get("TEXTREC_INK_THRESH", "160"))
# 墨迹垂直归一前，先在上下各多取这么多像素找墨迹（防止紧框本身已经把字形切掉）。
INK_PROBE_PAD = int(os.environ.get("TEXTREC_INK_PROBE_PAD", "4"))

# ★★ 2026-10-08 重做（第一版用 `dark.any(axis=1)` 判墨迹，被**横穿的导线**打穿：
#   导线是长直线，它经过的每一行都「有墨」，于是整块裁剪都被当成字，归一化等于
#   没做，实测 0.197 → 0.146 反而更差）。这一版按**形状**把线剔除：
#     · 某行里最长的连续暗游程 > `INK_RUN_FRAC × 裁剪宽`  ⇒ 这行是横线/导线，不算字；
#     · 从裁剪中心行上下扩张找字的上下边界，遇到 `INK_EMPTY_RUN` 行**连续全空**才停
#       （横线行是「有墨但被判为线」，可以穿过，所以不会把一条字切成两半；
#         而两行连续全空就是行间空白 ⇒ 不会把邻行的字并进来）。
#   归一目标不是「让字形占满 32px」，而是**复刻训练侧的裁剪**：训练框是紧标注框，
#   线上框是 stage1 的紧框 + `pad_box` 外扩。把上下边界都钉回「墨迹 ± PAD_Y_FRAC×
#   墨迹高」，两种框就得到同一张裁剪 —— 所以**不需要重训**。
INK_RUN_FRAC = float(os.environ.get("TEXTREC_INK_RUN_FRAC", "0.5"))
INK_MIN_FRAC = float(os.environ.get("TEXTREC_INK_MIN_FRAC", "0.10"))
INK_EMPTY_RUN = int(os.environ.get("TEXTREC_INK_EMPTY_RUN", "2"))
# 判定「长直线」的长度比例：形态学开运算的核长 = 该比例 × 带宽/带高（只留得下贯穿的线）。
INK_LINE_FRAC = float(os.environ.get("TEXTREC_INK_LINE_FRAC", "0.85"))
# 探针高度也可以按框高走：pp = max(INK_PROBE_PAD, INK_PROBE_PAD_FRAC × 框高)。
INK_PROBE_PAD_FRAC = float(os.environ.get("TEXTREC_INK_PROBE_PAD_FRAC", "0.25"))
# ★ 归一后的取景比例：裁剪高 = INK_FRAME × 墨迹高。2.5 对应「墨迹占裁剪高 0.40」，
#   正是训练侧 `g / (1.5×标注框高)` 的中位数。实测 g/H 的 p5..p95 = 0.47..1.36，
#   不是常数 ⇒ 只能对齐**取景比例**，没法反推出精确的标注框。
INK_FRAME = float(os.environ.get("TEXTREC_INK_FRAME", "2.5"))

# ---------------- 划分 ----------------
VAL_EVERY = 25        # 合成 case 每隔多少个抽一个进 val（按 case 抽，不是按裁剪抽）

# ---------------- 训练默认值 ----------------
DEFAULTS = dict(
    epochs=40,
    batch=256,
    lr=3e-4,
    weight_decay=1e-4,
    workers=6,
    seed=0,
    val_every=1,
    patience=10,
    grad_clip=5.0,
    min_freq=20,      # 字符表的最小频次门槛
)
