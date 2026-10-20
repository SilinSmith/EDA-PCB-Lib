# -*- coding: utf-8 -*-
"""推理接口——管线直接调这个。

    from textrec.infer import Recognizer
    rec = Recognizer.load(r"Y:\workspace\EDA\eda-pcb\textrec\runs\best.pt")

    img = Image.open(png_path)
    text, conf = rec.read(img, [x1, y1, x2, y2], role="designator")

    # 一张图上有很多文字行时，一次批量跑完比逐条快一个数量级：
    results = rec.read_many(img, [(bbox1, "designator"), (bbox2, "pin_number"), ...])

`bbox` 用**左上角原点的图像像素**，和数据集、和 `pipeline/` 里的约定一致。

`conf` 是**非空白字符的对数概率的几何平均**（即每个字平均有多确定），
取值 (0,1]。它是给下游做门槛用的：位号读错一个字符会连带丢掉该组件、
它的引脚、以及挂在它身上的所有网络，所以宁可拒识也不要误读——
用 `conf` 卡一个阈值，比让管线吞下错字符串安全得多。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from . import config as C
from .charset import BLANK, Charset
from .dataset import crop_gray
from .model import CRNN

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 宽度下采样倍数：CNN 里三次 MaxPool2d(2,2) 把宽度压到 1/8（见 model.py 的注释）。
WIDTH_REDUCTION = int(getattr(C, "WIDTH_REDUCTION", 8) or 8)

# 是否按裁片宽度分桶后再前向（默认**关**）。
#
# ⚠️ 实测判负，不要开：分桶让补白归零，等价于「无补白前向」。但训练侧的
# `collate` 是把整批补到本批最大宽度，模型习惯了「左侧内容 + 右侧大片白」，
# 喂无补白的裁片反而是分布外。快评（`work\tools\diag\_crnneval.py`）：
#     批内最大补白 + 全 T 解码      exact 0.6815  位号 0.8407
#     分桶（无补白）                （见 b319 的 220 张端到端：23.8329 -> 23.6422）
BUCKET_BY_WIDTH = os.environ.get("TEXTREC_BUCKET", "0").strip() != "0"

# 解码时在 `w // WIDTH_REDUCTION` 之后多留几个时步（默认 0，即严格对齐训练侧的
# `input_len`）。留正数只在实验里有意义。
EXTRA_STEPS = int(os.environ.get("TEXTREC_EXTRA_STEPS", "0") or 0)

Bbox = Sequence[float]
ImageLike = Union[str, Path, Image.Image, np.ndarray]


def _as_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image if image.mode == "L" else image.convert("L")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("L")
    if isinstance(image, np.ndarray):
        a = image
        if a.ndim == 3:
            a = a[..., :3].mean(axis=2)
        return Image.fromarray(a.astype("uint8"), mode="L")
    raise TypeError("不认识的图像类型: %r" % type(image))


def _role_id(role) -> int:
    if isinstance(role, int):
        return max(0, min(C.N_ROLES - 1, role))
    return C.ROLE_ID.get(str(role), C.N_ROLES - 1)


def normalize_device(device) -> str:
    """把「ultralytics 风格」的设备写法翻译成 `torch.device` 认识的字符串。

    ⚠️ 这是提交容器里踩过的**致命坑**：ultralytics / YOLO 接受 `device="0"`、
    `"0,1"`、`"cpu"`；但 `torch.device("0")` 会直接抛
    `RuntimeError: Invalid device string: '0'`。以前 `TextEngine` 拿到外面的
    `-device 0` 就在静默中加载失败（`allow_missing=True`），**全部位号的 OCR
    结果变成空** —— 表面上流水线「跑通了」，实际 component F1 掉到 0。

    在这里统一收口，所有调用点（stage2 装配、stage3、提交驱动）都因此安全。
    """
    d = str(device if device is not None else "auto").strip()
    low = d.lower()
    if d == "" or low == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if low in ("cpu", "-1", "none", "mps"):
        return "cpu" if low != "mps" else "mps"
    if low.startswith("cuda"):
        return d
    if d[0].isdigit():                       # "0" / "0,1" / "0:1" -> 取第一张卡
        if not torch.cuda.is_available():
            return "cpu"                     # 没卡就安静退到 CPU，而不是炸掉整批
        return "cuda:" + d.split(",")[0].strip()
    return d


class Recognizer:
    def __init__(self, model: CRNN, charset: Charset, device: torch.device,
                 img_h: int = None, max_w: int = None):
        self.model = model.to(device).eval()
        self.charset = charset
        self.device = device
        self.img_h = img_h or C.IMG_H
        self.max_w = max_w or C.MAX_W
        # 同一个宽度桶内一次前向的样本数上限（桶内宽度全等，补白为 0）。
        self.bucket_max = max(1, int(os.environ.get("TEXTREC_BUCKET_MAX", "64")))

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, ckpt: Union[str, Path], device: str = "auto") -> "Recognizer":
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError("找不到权重: %s" % ckpt)
        dev = torch.device(normalize_device(device))

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        charset = Charset(ck["charset"])
        cfg = ck.get("config", {}) or {}
        model = CRNN(charset.num_classes)
        model.load_state_dict(ck["model"])
        r = cls(model, charset, dev,
                img_h=cfg.get("img_h", C.IMG_H), max_w=cfg.get("max_w", C.MAX_W))
        r.ckpt_meta = {k: v for k, v in ck.items()
                       if k not in ("model", "opt", "charset")}
        return r

    # ------------------------------------------------------------------
    def _decode(self, col_ids, lps) -> Tuple[str, float]:
        """CTC 贪心折叠。`lps[t]` 是第 t 个时步整行对数概率，取命中类的那一项。"""
        chars, keep = [], []
        prev = None
        for t, i in enumerate(col_ids):
            if i != prev and i != BLANK:
                ch = self.charset.itos.get(i)
                if ch is not None:
                    chars.append(ch)
                    keep.append(lps[t][i])
            prev = i
        conf = float(np.exp(np.mean(keep))) if keep else 0.0
        return "".join(chars), conf

    @torch.no_grad()
    def read_many(self, image: ImageLike,
                  items: Iterable[Tuple[Bbox, object]]) -> List[Tuple[str, float]]:
        """items: [(bbox, role), ...] -> [(text, conf), ...]

        ⚠️ **按裁片宽度分桶可消除补白**（`BUCKET_BY_WIDTH`，由
        `TEXTREC_BUCKET` 控制，**当前默认 `0`＝关**）。

        默认关是有意的：下面 `_forward_padded` 已经**按每个样本自己的有效时步
        `w // WIDTH_REDUCTION` 截断**，补白那段的输出不再参与解码，所以补白
        对结果只剩「卷积感受野从右侧渗进来」这一点点影响。实测（10GT case 1
        的 6 个位号框 `t0027..t0031,t0024`）：
        `TEXTREC_BUCKET=0` → `C51/C15/C5/C54/C5/C15`；
        `TEXTREC_BUCKET=1` → `C5.1/C15/C5/C54/C5/C15`
        —— 只有 1 个框的读法不同，且两个都是错的（真值 `C151..C156`）。
        ⇒ **这些错不是补白造成的，是模型本身读错**，分桶不能救。

        原来这里把整批补白到 `max_w = max(widths)` 再解码全部 T 个时步。
        训练侧的 `collate` 补白方式完全相同，但 **CTC 损失用的是
        `ctc_input_lens_for(widths, T)`（即 `w // WIDTH_REDUCTION`）把补白那段
        屏蔽掉了**，而这里的贪心解码没有屏蔽 —— 补白会让模型凭空多吐字符。
        实测（case 0001，539 个文字框）：同一批里混进一条宽裁片，就有
        **462/539 = 85.7%** 的读法与逐条读取不同，且几乎全部是**多出尾字符**
        （`R2 -> R2441`、`L2 -> L21`、`1 -> 111`、`C4 -> C44`）。
        分桶后同一桶内宽度全等 ⇒ 补白为零 ⇒ **与逐条读取逐字节一致**，
        而调用次数只从 1 次变成「不同宽度的个数」次（case 0001 是 96 次）。
        """
        img = _as_pil(image)
        items = list(items)
        if not items:
            return []

        arrays, widths, roles = [], [], []
        for bbox, role in items:
            crop = crop_gray(img, bbox, self.img_h, self.max_w)
            a = np.asarray(crop, dtype=np.float32) / 255.0
            a = (a - 0.5) / 0.5
            arrays.append(a.astype(np.float32))
            widths.append(a.shape[1])
            roles.append(_role_id(role))

        if not BUCKET_BY_WIDTH:
            return self._forward_padded(arrays, widths, roles)

        # 按宽度严格分桶：桶内宽度全等，补白为 0。
        buckets: "dict[int, list[int]]" = {}
        for i, w in enumerate(widths):
            buckets.setdefault(w, []).append(i)

        out: List[Tuple[str, float]] = [("", 0.0)] * len(items)
        for w, idxs in buckets.items():
            for s in range(0, len(idxs), self.bucket_max):
                part = idxs[s:s + self.bucket_max]
                batch = torch.from_numpy(
                    np.stack([arrays[i] for i in part])[:, None, :, :]
                ).to(self.device)
                role_t = torch.tensor([roles[i] for i in part],
                                      dtype=torch.long, device=self.device)
                logp = self.model(batch, role_t).log_softmax(2)   # T,B,C
                idx = logp.argmax(2)                              # T,B
                # 有效时步 = w // WIDTH_REDUCTION（与训练侧 input_len 同式）；
                # 桶内宽度全等，所以这里恒等于 T，截断是恒等变换，留着只为防御。
                tmax = max(1, min(idx.shape[0], w // WIDTH_REDUCTION))
                for k, i in enumerate(part):
                    out[i] = self._decode(idx[:tmax, k].tolist(),
                                          logp[:tmax, k].tolist())
        return out

    @torch.no_grad()
    def _forward_padded(self, arrays, widths, roles) -> List[Tuple[str, float]]:
        """整批补白到本批最大宽度（与训练侧 `collate` 完全一致），**但解码只取
        每个样本自己的有效时步** `w // WIDTH_REDUCTION`（与训练侧 CTC 的
        `input_len = ctc_input_lens_for(widths, T)` 同式）。

        ⚠️ 这里以前解码全部 T 个时步，把补白那段也算进去了。补白在**训练时被
        CTC 的 `input_lengths` 屏蔽**，模型对它没有任何监督约束，于是它吐出什么
        全靠猜，而且**猜出来的字符会连在真结果后面**。快评
        （`work\\tools\\diag\\_crnneval.py`，训练用的 val 785 条）：

            批大小 32， 全 T 解码   exact 0.6815   位号 0.8407
            批大小 32， 截断        exact 0.6879   位号 0.8761
            批大小 256（= 线上 `CHUNK`），全 T 解码  exact 0.6573  位号 0.7522
            批大小 256，截断                          exact 0.6866  位号 0.8850

        ⇒ **批越大、补白越宽，全 T 解码错得越狠**（位号 0.8407 → 0.7522），
        而截断把这个损失整个收回来（位号 **+13.3pp**，且批大小 256 反而最好）。
        这就是线上 `R2 -> R2441`、`C4 -> C44`、`1 -> 111` 这批错的来源。
        """
        max_w = max(widths)
        h = arrays[0].shape[0]
        batch = torch.full((len(arrays), 1, h, max_w), 1.0, dtype=torch.float32)
        for i, a in enumerate(arrays):
            batch[i, :, :, : a.shape[1]] = torch.from_numpy(a)
        batch = batch.to(self.device)
        role_t = torch.tensor(roles, dtype=torch.long, device=self.device)
        logp = self.model(batch, role_t).log_softmax(2)
        idx = logp.argmax(2)
        T = idx.shape[0]
        out = []
        for b in range(len(arrays)):
            tmax = max(1, min(T, widths[b] // WIDTH_REDUCTION + EXTRA_STEPS))
            out.append(self._decode(idx[:tmax, b].tolist(), logp[:tmax, b].tolist()))
        return out

    # ------------------------------------------------------------------
    def read(self, image: ImageLike, bbox: Bbox,
             role: object = "designator") -> Tuple[str, float]:
        return self.read_many(image, [(bbox, role)])[0]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def read_topk(self, image: ImageLike, items: Iterable[Tuple[Bbox, object]],
                  k: int = 5, beam: int = 8) -> List[List[Tuple[str, float]]]:
        """每个框返回至多 `k` 条候选 `[(text, conf), ...]`，按整条路径概率降序。

        贪心解码只保留 argmax 那一条；而位号错例里最常见的正是「差 1 个字符」
        （`U202 -> U22`、`C157 -> C19`），正确答案往往就是同一框的第 2/3 条假设。
        这里用 CTC 前缀 beam 搜索把它们留住，交给下游按全页一致性挑选。

        `conf` 口径与 `read_many` 相同（被发射字符的对数概率几何平均），因此下游
        的门槛不用改。逐条前向（不补白），保证同一框的候选不受同批其它样本影响。
        """
        img = _as_pil(image)
        items = list(items)
        if not items:
            return []
        out: List[List[Tuple[str, float]]] = []
        for bbox, role in items:
            crop = crop_gray(img, bbox, self.img_h, self.max_w)
            a = np.asarray(crop, dtype=np.float32) / 255.0
            a = ((a - 0.5) / 0.5).astype(np.float32)
            batch = torch.from_numpy(a[None, None, :, :]).to(self.device)
            role_t = torch.tensor([_role_id(role)], dtype=torch.long,
                                  device=self.device)
            logp = self.model(batch, role_t).log_softmax(2)[:, 0, :].cpu().numpy()
            tmax = max(1, min(logp.shape[0], a.shape[1] // WIDTH_REDUCTION + EXTRA_STEPS))
            out.append(self._beam(logp[:tmax], k=k, beam=beam))
        return out

    def _beam(self, lp, k: int = 5, beam: int = 8) -> List[Tuple[str, float]]:
        """CTC 前缀 beam 搜索。状态 = (前缀, log p(blk), log p(非blk), 字数和, 字数)。"""
        neg = -1e9
        beams = {(): (0.0, neg, 0.0, 0)}
        for t in range(lp.shape[0]):
            row = lp[t]
            nxt = {}
            cand = np.argpartition(-row, min(beam, row.shape[0] - 1))[:beam]
            for pre, (pb, pnb, ssum, n) in beams.items():
                tot = np.logaddexp(pb, pnb)
                a = nxt.get(pre, (neg, neg, 0.0, 0))
                nxt[pre] = (np.logaddexp(a[0], tot + float(row[BLANK])),
                            a[1], a[2], a[3])
                for c in cand:
                    c = int(c)
                    if c == BLANK:
                        continue
                    p = float(row[c])
                    if pre and pre[-1] == c:
                        a = nxt.get(pre, (neg, neg, 0.0, 0))
                        nxt[pre] = (a[0], np.logaddexp(a[1], pnb + p),
                                    a[2] + p, a[3] + 1)
                        key = pre + (c,)
                        a = nxt.get(key, (neg, neg, 0.0, 0))
                        nxt[key] = (a[0], np.logaddexp(a[1], tot + p),
                                    a[2] + p, a[3] + 1)
                    else:
                        key = pre + (c,)
                        a = nxt.get(key, (neg, neg, 0.0, 0))
                        nxt[key] = (a[0], np.logaddexp(a[1], tot + p),
                                    a[2] + p, a[3] + 1)
            beams = dict(sorted(nxt.items(),
                                key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))[:beam])
        ranked = sorted(beams.items(),
                        key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))[:k]
        out = []
        for pre, (pb, pnb, ssum, n) in ranked:
            txt = "".join(self.charset.itos.get(i, "") for i in pre)
            out.append((txt, float(np.exp(ssum / n)) if n else 0.0))
        return out

    # ------------------------------------------------------------------
    def debug(self) -> str:
        meta = getattr(self, "ckpt_meta", {})
        return ("Recognizer(device=%s, classes=%d, img_h=%d, max_w=%d, "
                "val_cer=%s, val_exact=%s)"
                % (self.device, self.charset.num_classes, self.img_h, self.max_w,
                   meta.get("val_cer"), meta.get("val_exact")))
