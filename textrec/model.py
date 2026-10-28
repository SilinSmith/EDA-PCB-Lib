# -*- coding: utf-8 -*-
"""CRNN + CTC 识别模型。

结构：CNN（宽度下采样 8 倍、高度压到 1）-> 拼上角色嵌入 -> 双向 LSTM -> 线性头 -> CTC。

为什么是 CRNN 而不是别的
------------------------
* **不需要切字**。语料里的位号紧贴（`U1`、`R10`），切格错误是现有管线失败的直接原因
  （`docs/算法路线设计.md` 5.4：`ccells=0.476` 是任何分割器的上界，而 `exact` 只有 0.207）。
  CTC 直接对整行建模，绕开切格。
* **字符串长度天然可变**（1~24 字符），不需要固定长度分类头。
* **角色条件是免费的**：调用方（文本行检测器）已经知道这一段是位号还是引脚号，
  把它作为嵌入拼进每个时间步，能让同一套权重在不同角色上表现得像专门模型。

宽度下采样固定为 8（对应 `config.WIDTH_REDUCTION`）。语料里最长的字符串是 16 字符，
其宽高比约 7.2，缩到高 32 后宽约 230，时间步 T≈28，足够 CTC 表达。
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from . import config as C

# 见 CRNN.forward 里的说明：默认不用 cuDNN 跑 RNN。
_RNN_CUDNN_OK = os.environ.get("TEXTREC_CUDNN_RNN") == "1"

_CONV_STAGES = ((1, 64), (64, 128), (128, 256), (256, 256), (256, 512), (512, 512))
# 池化落在哪些卷积段之后。第 3 段（128→256）不池化，与 2026-10 之前写死的结构一致
# （那一段是「两个卷积 + 一次池化」）。多余的高度池化接在末尾。
_POOL_SLOTS = (0, 1, 3, 4, 5)


def _pool_plan(img_h: int, width_reduction: int):
    """返回 (每个卷积段之后的 (高, 宽) 池化因子, 末尾追加的 (高, 宽))。

    高度因子连乘 = img_h，宽度因子连乘 = width_reduction。
    默认 `IMG_H=32 / WIDTH_REDUCTION=8` 得到
    stage0 (2,2) / stage1 (2,2) / stage3 (2,2) / stage4 (2,1) / stage5 (2,1)，
    与旧结构逐层一致 ⇒ `weights/textrec_best.pt` 可直读。
    """
    n_h = int(round(math.log2(img_h)))
    n_w = int(round(math.log2(width_reduction)))
    if 2 ** n_h != img_h or 2 ** n_w != width_reduction:
        raise ValueError("IMG_H / WIDTH_REDUCTION 必须是 2 的幂：%r / %r"
                         % (img_h, width_reduction))
    per_stage = [(1, 1)] * len(_CONV_STAGES)
    for j, si in enumerate(_POOL_SLOTS):
        per_stage[si] = (2 if j < n_h else 1, 2 if j < n_w else 1)
    tail = [1, 1]
    for extra in range(n_h - len(_POOL_SLOTS)):
        if extra == 0:
            tail = [2, 1]
        else:
            raise ValueError("池化槽位不足：IMG_H=%d 需要更多高度池化" % img_h)
    return per_stage, tuple(tail)


def _cnn_layers():
    per_stage, tail = _pool_plan(C.IMG_H, C.WIDTH_REDUCTION)
    layers = []
    for (cin, cout), (hf, wf) in zip(_CONV_STAGES, per_stage):
        layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True)]
        if hf > 1 or wf > 1:
            layers.append(nn.MaxPool2d((hf, wf), (hf, wf)))
    if tail[0] > 1 or tail[1] > 1:
        layers.append(nn.MaxPool2d(tail, tail))
    return layers


class CRNN(nn.Module):
    def __init__(self, num_classes: int, n_roles: int = C.N_ROLES,
                 rnn_hidden: int = 256, role_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.n_roles = n_roles

        self.cnn = nn.Sequential(*_cnn_layers())
        # 到这里高度已经被压成 1（IMG_H 与 WIDTH_REDUCTION 都是 2 的幂，
        # 池化层数由 `_pool_plan` 按 config 生成；默认 32/8 与旧结构逐层一致）。

        self.role_emb = nn.Embedding(n_roles, role_dim)
        self.rnn = nn.LSTM(512 + role_dim, rnn_hidden, num_layers=2,
                           bidirectional=True, batch_first=False,
                           dropout=dropout if dropout > 0 else 0.0)
        self.head = nn.Linear(2 * rnn_hidden, num_classes)

    def forward(self, x: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W)  role: (B,)  ->  logits: (T,B,num_classes)"""
        feat = self.cnn(x)                       # B, 512, 1, T
        b, c, h, w = feat.shape
        if h != 1:
            feat = feat.mean(dim=2)              # 兜底：万一高度没压干净
            w = feat.shape[-1]
        feat = feat.reshape(b, c, w).permute(2, 0, 1)      # T, B, 512

        remb = self.role_emb(role).unsqueeze(0)            # 1, B, role_dim
        remb = remb.expand(feat.shape[0], -1, -1)          # T, B, role_dim
        feat = torch.cat([feat, remb], dim=2)              # T, B, 512+role_dim

        # 这一行必须绕开 cuDNN，否则进程退出时会被 Windows 的 DLL 卸载流程
        # fail-fast 掉（退出码 -1073740791 = 0xC0000409 STATUS_STACK_BUFFER_OVERRUN）。
        # 实测（RTX 4080，batch 64，宽 112）：
        #     cuDNN 全程参与   45.0 ms/step   退出码 -1073740791
        #     只 RNN 走原生    79.1 ms/step   退出码 0
        #     全局关 cuDNN    168.4 ms/step   退出码 0
        # 训练产物（best.pt/last.pt/history.json）在崩与不崩两种情况下都是完整的，
        # 但退出码非零会让人误判训练失败；`os._exit(0)` 也躲不开，因为崩在
        # DLL_PROCESS_DETACH。关掉 TEXTREC_CUDNN_RNN=1 可以换回 cuDNN 版 RNN
        # （更快，但退出码非零）。
        if _RNN_CUDNN_OK:
            out, _ = self.rnn(feat)                        # T, B, 2*hidden
        else:
            with torch.backends.cudnn.flags(enabled=False):
                out, _ = self.rnn(feat)                    # T, B, 2*hidden
        return self.head(out)                              # T, B, num_classes

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy(self, x: torch.Tensor, role: torch.Tensor, charset) -> "list[str]":
        logits = self.forward(x, role)                     # T, B, C
        ids = logits.argmax(dim=2).permute(1, 0)           # B, T
        return [charset.greedy(row.tolist()) for row in ids]


def build(num_classes: int, **kw) -> CRNN:
    m = CRNN(num_classes, **kw)
    n = sum(p.numel() for p in m.parameters())
    m._n_params = n                                        # type: ignore[attr-defined]
    return m
