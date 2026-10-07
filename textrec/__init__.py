# -*- coding: utf-8 -*-
"""textrec —— 原理图文字识别器。

这一包解决的是赛题六里**性价比最高的一段**：把位号（`R1`/`U1`/`C87`）从图上读出来。

背景（见 `docs/算法路线设计.md` 第 4.2 节）：
    现有管线在 `read-key+skel` 这一级的端到端得分是 **恰好 0.0000**，而全部
    四条 read* 变体都是 0。原因不在检测、不在骨架——文字行都找到了（0025 找到
    19 行、0036 找到 32 行、0059 找到 21 行），而是**模板匹配这一步既拒识又误读**，
    于是一个位号都没读出来，`read_components` 拿不到位号就跳过，组件集合为空，
    component / pin / net 三块同时归零。

本包的训练数据是现成的：`合成原理图数据集/texts.jsonl` 里有 **381876 条**文字实例，
带 `content`（字符串）、`role`（六类角色）、`bbox`（左上角原点像素）。不需要再标注。

典型用法：
    # 1) 建清单 + 字符表
    python -m textrec.manifest

    # 2) 训练
    python -m textrec.train --epochs 40 --batch 256

    # 3) 真实语料上验一次（唯一诚实的泛化检查）
    python -m textrec.eval_real

    # 4) 在管线里用
    from textrec.infer import Recognizer
    rec = Recognizer.load("textrec/runs/best.pt")
    text, conf = rec.read(pil_image, bbox, role="designator")
"""

__all__ = ["config", "charset", "manifest", "dataset", "model", "train", "infer"]
