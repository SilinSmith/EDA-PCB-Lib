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

import os

import torch
import torch.nn as nn

from . import config as C

# 见 CRNN.forward 里的说明：默认不用 cuDNN 跑 RNN。
_RNN_CUDNN_OK = os.environ.get("TEXTREC_CUDNN_RNN") == "1"


class CRNN(nn.Module):
    def __init__(self, num_classes: int, n_roles: int = C.N_ROLES,
                 rnn_hidden: int = 256, role_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.n_roles = n_roles

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                    # H/2,  W/2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                    # H/4,  W/4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                                    # H/8,  W/8
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                          # H/16, W/8
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                          # H/32, W/8
        )
        # 到这里高度已经被压成 1（要求 IMG_H 是 32 的倍数）。

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
