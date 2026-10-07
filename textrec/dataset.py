# -*- coding: utf-8 -*-
"""裁剪数据集：把 manifest 里的 bbox 变成固定高度的灰度张量。

设计要点
--------
* **不预先落盘裁剪图**。381876 条裁剪写成 PNG 又慢又占地方，而且同一个 case 的
  所有文字都在同一张图上——这里改成运行时裁剪 + 每 worker 一个图片 LRU 缓存。
* **缩放到固定高度、宽度按长宽比**。语料实测：pin_number 高 4.3 px（恒定）、
  pin_name 5.0、designator/value 5.7、net_label 6.7。也就是统一高度 32 意味着
  **放大约 6~7 倍**，这正是识别小字的关键（原图那个尺度人眼也读不出）。
* **角色作为条件输入**。六类角色的字符集和长度分布差异极大（pin_number 基本是
  纯数字、长度 ≤3；value 有 "crystal_2pin"），把 role 喂给模型是白捡的准确率。
* 宽度不足时补 1.0（白底），与归一化后的背景取值一致。
"""
from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

from . import config as C
from .charset import Charset


def _ink_band(img: Image.Image, cy1: int, cy2: int, cx1: int, cx2: int,
              W: int, H: int, augment: bool = False, rng=None):
    """在裁剪上下各多探一段像素，找出**字**的上下边界，并直接给出归一后的裁剪。

    返回 ``(y1, y2, glyph_h)``（图像坐标、浮点），找不到可靠的字就返回 ``None``。

    三步（每一版都比上一版更稳，见 config 里 INK_* 的注释）：
      ① **先把长直线抠掉**。导线 / 图框是贯穿整条带的笔画，用形态学开运算
         （核长 = `INK_LINE_FRAC × 带宽`）只留得下贯穿的线，字的笔画全被抹掉；
         再从暗像素里减掉这些线，剩下的就是字。
         —— 这一步是关键：第一版用 `dark.any(axis=1)` 判墨迹，横穿的导线让每一行
         都「有墨」，归一化等于没做（0.197 → 0.146）；第二版改用「最长暗游程占框宽
         的比例」，又把**短标签自己的笔画**（`1` 的底衬线、`2` 的底横）误判成线，
         把字裁掉一截（designator 0.84 → 0.36）。形态学版对两者都免疫。
      ② 从裁剪中心行上下扩张找字的上下边界：可以穿过「被抠掉的线」所在的行走，
         遇到 `INK_EMPTY_RUN` 行连续全空才停（那是行间空白），所以不会把邻行的字
         并进来、也不会把一条字切成两半。
      ③ 按 `INK_FRAME × 墨迹高` 定裁剪高度，居中套在墨迹上。

    ★ 归一的目标**不是**「让字形占满 32px」，而是把线上那张裁剪拉回训练分布。
    训练侧 `crop_gray` 拿到的框是标注框 H，裁剪高 = `(1 + 2×PAD_Y_FRAC) × H`；
    实测 `墨迹高g / 标注框高H` 在 p5..p95 是 0.47..1.36（**不是常数**，标注框的松紧
    本来就参差），所以没法反推出精确的 H；能做的只是让**取景比例**落回训练分布的
    中位数：训练侧 `g / 裁剪高` 的中位约 0.40 ⇒ `INK_FRAME = 1/0.40 ≈ 2.5`。
    """
    h0 = max(1.0, float(cy2 - cy1))
    pp = max(C.INK_PROBE_PAD, int(round(C.INK_PROBE_PAD_FRAC * h0)))
    if augment and rng is not None:
        pp = int(round(pp * rng.uniform(0.5, 1.0)))
    py1 = int(max(0, cy1 - pp))
    py2 = int(min(H, cy2 + pp))
    if py2 - py1 < 4 or cx2 - cx1 < 2:
        return None
    band = np.asarray(img.crop((cx1, py1, cx2, py2)).convert("L"), dtype=np.uint8)
    if band.ndim != 2 or band.shape[0] < 4 or band.shape[1] < 2:
        return None
    dark = ((band < C.INK_THRESH).astype(np.uint8)) * 255
    if not dark.any():
        return None
    bh, bw = dark.shape

    # ① 抠长直线：横线与竖线各做一次开运算，再膨胀 1px 抹掉边缘残留。
    kx = max(3, int(round(C.INK_LINE_FRAC * bw)))
    ky = max(3, int(round(C.INK_LINE_FRAC * bh)))
    line = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, kx), np.uint8))
    if ky < bh:
        line = cv2.bitwise_or(
            line, cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((ky, 1), np.uint8)))
    line = cv2.dilate(line, np.ones((3, 3), np.uint8))
    glyph = cv2.bitwise_and(dark, cv2.bitwise_not(line))

    row_g = (glyph > 0).any(axis=1)       # 这一行有「字」
    row_d = (dark > 0).any(axis=1)        # 这一行有暗像素（含线）
    if not row_g.any():
        return None
    nrow = bh

    # ② 从中心行上下扩张。
    mid = int(round(0.5 * (cy1 + cy2))) - py1
    mid = max(0, min(nrow - 1, mid))
    if not row_g[mid]:
        cand = np.flatnonzero(row_g)
        mid = int(cand[np.argmin(np.abs(cand - mid))])
    a = b = mid
    empty = 0
    while a - 1 >= 0:
        if row_g[a - 1]:
            a -= 1
            empty = 0
        elif not row_d[a - 1]:
            empty += 1
            if empty >= C.INK_EMPTY_RUN:
                break
            a -= 1
        else:                              # 被抠掉的线所在的行，穿过去
            a -= 1
            empty = 0
    empty = 0
    while b + 1 < nrow:
        if row_g[b + 1]:
            b += 1
            empty = 0
        elif not row_d[b + 1]:
            empty += 1
            if empty >= C.INK_EMPTY_RUN:
                break
            b += 1
        else:
            b += 1
            empty = 0
    ta = np.flatnonzero(row_g[a:b + 1])
    if ta.size == 0:
        return None
    a = a + int(ta[0])
    b = a + int(ta[-1] - ta[0])
    glyph_h = float(b - a + 1)
    if not (2.0 <= glyph_h <= 0.95 * nrow):
        return None

    # ③ 以墨迹为中心套一个 `INK_FRAME × glyph_h` 高的框。
    top = float(py1 + a)
    bot = float(py1 + b + 1)
    cyc = 0.5 * (top + bot)
    ch = max(4.0, C.INK_FRAME * glyph_h)
    ny1 = int(max(0, min(H - 1, round(cyc - 0.5 * ch))))
    ny2 = int(max(ny1 + 1, min(H, round(cyc + 0.5 * ch))))
    if ny2 - ny1 < 4:
        return None
    return ny1, ny2, glyph_h


def crop_gray(img: Image.Image, bbox, img_h: "int | None" = None,
              max_w: "int | None" = None, augment: bool = False, rng=None):
    """按 bbox 裁一块灰度图并缩放到统一高度。**训练与推理必须共用这一个函数**，
    否则预处理不一致会让线上精度无端掉一截。"""
    img_h = img_h or C.IMG_H
    max_w = max_w or C.MAX_W
    rng = rng or random

    x1, y1, x2, y2 = [float(v) for v in bbox]
    h = max(1.0, y2 - y1)
    mx = C.PAD_X_FRAC * h
    my = C.PAD_Y_FRAC * h
    if augment:
        # 轻微抖动：位置 ±1px、边距 ±30%。文字本来就小，抖动幅度要克制。
        mx *= rng.uniform(0.7, 1.4)
        my *= rng.uniform(0.7, 1.4)
        x1 += rng.uniform(-1.0, 1.0)
        y1 += rng.uniform(-0.6, 0.6)
        x2 += rng.uniform(-1.0, 1.0)
        y2 += rng.uniform(-0.6, 0.6)

    W, H = img.size
    cx1 = int(max(0, min(W - 1, round(x1 - mx))))
    cy1 = int(max(0, min(H - 1, round(y1 - my))))
    cx2 = int(max(cx1 + 1, min(W, round(x2 + mx))))
    cy2 = int(max(cy1 + 1, min(H, round(y2 + my))))

    if C.INK_YNORM:
        # 把裁剪的垂直范围对齐到墨迹，取景比例拉回训练分布（见 _ink_band 的 docstring）。
        try:
            got = _ink_band(img, cy1, cy2, cx1, cx2, W, H, augment, rng)
        except Exception:                                        # noqa: BLE001
            got = None
        if got is not None:
            cy1, cy2, _gh = got

    crop = img.crop((cx1, cy1, cx2, cy2))

    ch, cw = crop.size[1], crop.size[0]
    new_w = int(round(img_h * cw / max(1, ch)))
    new_w = max(C.MIN_W, min(max_w, new_w))
    return crop.resize((new_w, img_h), Image.BILINEAR)


class CropDataset(Dataset):
    def __init__(self, records: Sequence[dict], charset: Charset,
                 img_h: int = None, max_w: int = None,
                 augment: bool = False, cache_size: int = 64,
                 drop_unk: bool = True):
        self.charset = charset
        self.img_h = img_h or C.IMG_H
        self.max_w = max_w or C.MAX_W
        self.augment = augment
        self.cache_size = max(1, cache_size)
        self._cache: "OrderedDict[str, Image.Image]" = OrderedDict()
        self.n_dropped_unk = 0

        kept: List[dict] = []
        for r in records:
            tgt = charset.try_encode(r["text"])
            if tgt is None:
                self.n_dropped_unk += 1
                if drop_unk:
                    continue
                tgt = []
            r = dict(r)
            r["_target"] = tgt
            kept.append(r)
        self.records = kept

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def _image(self, path: str) -> Image.Image:
        img = self._cache.get(path)
        if img is not None:
            self._cache.move_to_end(path)
            return img
        img = Image.open(path).convert("L")
        self._cache[path] = img
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img

    # ------------------------------------------------------------------
    def _crop(self, img: Image.Image, bbox) -> Image.Image:
        return crop_gray(img, bbox, self.img_h, self.max_w,
                         augment=self.augment, rng=random)

    # ------------------------------------------------------------------
    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        img = self._image(r["image"])
        crop = self._crop(img, r["bbox"])
        a = np.asarray(crop, dtype=np.float32) / 255.0
        if self.augment:
            # 亮度/对比度抖动。原理图白底黑字，这里只是让模型别把绝对灰度当特征。
            a = np.clip((a - 0.5) * random.uniform(0.85, 1.15) + 0.5
                        + random.uniform(-0.06, 0.06), 0.0, 1.0)
        a = (a - 0.5) / 0.5                      # 背景(白)=+1, 墨(黑)=-1
        x = torch.from_numpy(a).unsqueeze(0)     # 1, H, W
        return {
            "img": x,
            "target": torch.tensor(r["_target"], dtype=torch.long),
            "role": torch.tensor(int(r["role_id"]), dtype=torch.long),
            "text": r["text"],
            "case": r["case"],
            "src": r["src"],
        }


def collate(batch: List[dict]) -> dict:
    """把不同宽度的裁剪补到本批最大宽度。补 1.0（白底）。"""
    max_w = max(b["img"].shape[2] for b in batch)
    h = batch[0]["img"].shape[1]
    imgs = torch.full((len(batch), 1, h, max_w), 1.0, dtype=torch.float32)
    widths: List[int] = []
    for i, b in enumerate(batch):
        w = b["img"].shape[2]
        imgs[i, :, :, :w] = b["img"]
        widths.append(w)

    targets = torch.cat([b["target"] for b in batch]) if batch else torch.zeros(0, dtype=torch.long)
    target_lens = torch.tensor([len(b["target"]) for b in batch], dtype=torch.long)
    roles = torch.stack([b["role"] for b in batch])
    # CTC 的输入长度：有效宽度按同样的宽度下采样倍数折算。
    input_lens = torch.tensor(
        [max(1, w // C.WIDTH_REDUCTION) for w in widths], dtype=torch.long)
    return {
        "img": imgs,
        "target": targets,
        "target_len": target_lens,
        "input_len": input_lens,
        "role": roles,
        "widths": widths,
        "text": [b["text"] for b in batch],
        "case": [b["case"] for b in batch],
        "src": [b["src"] for b in batch],
    }


def ctc_input_lens_for(widths: Sequence[int], out_len: int):
    """把有效宽度折算成 CTC 的时间步长度，并夹到模型真实输出长度以内。"""
    import torch as _t
    return _t.tensor([max(1, min(out_len, w // C.WIDTH_REDUCTION)) for w in widths],
                     dtype=_t.long)
