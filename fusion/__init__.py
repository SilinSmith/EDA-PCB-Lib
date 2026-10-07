# -*- coding: utf-8 -*-
"""融合层：把**所有读取器**的证据汇到一个池子里，再用约束求解 + 带图的 LLM 判断去裁决。

存在理由（`work\\reports\\节点活性体检.md` 的实测结论）
----------------------------------------------------
`work\\flow\\workflow.html` 画的 22 个节点里，右侧「融合与仲裁」整列
（`pool` / `solve` / `plausible` / `conflict` / `llm_judge` / `rescan` / `iterate` / `know`）
**一个文件都没有** —— 图上那条 `pool.claims → solve.sol → assemble.sol` 的扇形汇入是
画出来的。实际在解的是 `stage2/assign.py` 的**局部贪心**（每颗元件各自取最优），
于是：

* 220 张实测 `stage3`（LLM 仲裁）与 `stage2` **逐位相同**（component tp/fp/fn 全等）
  —— 因为 `stage1/llm.py` 从头到尾**没有图片通道**（messages 是纯文本、全文件无
  `image_url`/`base64`），而 `arbitrate.py` 的 docstring 把这个「没接线」误诊成了
  「模型是纯文本的」；
* 唯一真正在干活的 LLM 节点是 `vlm/enrich.py`（读位号），它**不参与任何决策**，
  而且因为没人仲裁框的归属，它多制造了 **139 个**新的 `component.bbox_mismatch`
  （220 张：281 → 420）。

本包按 `workflow.html` 补齐这一列：

    claims.py     证据池：所有读取器 -> 统一 Slot/Alt（复用 cvparts.hypo 的代数）
    priors.py     常识与先验表：前缀↔类型、脚号阶梯、电源轨名、单位词表
    solve.py      约束求解 + 冲突与缺口 + 迭代控制
    plausible.py  全局合理性检查（结构）：一框一主、键唯一、引脚归属唯一
    llm_judge.py  带图的 LLM 判断（**这是补上图片通道的地方**）
    rescan.py     按优先级定向重扫
    iterate.py    定向回传：把 LLM 读到的 **引脚号/名** 写回池子，让 `stage2`
                  用真读数重新装配（`stage1\pincol.py` 枚举出来的框外「号码窗」与
                  框内「名字窗」就是它的输入）
    run.py        CLI：读 stage2 产物 -> 输出改写后的 `_assigned`

**哪些真的在干活（实测，2025-xx 的 19 张定点档）**

| 模块 | 状态 | 证据 |
|---|---|---|
| `claims` / `priors` / `solve` / `plausible` | 在跑 | 融合层每张图都过一遍 |
| `llm_judge` | 在跑，**中性** | 整表模式（`FUSION_JUDGE_MODE=sheet`，一次请求答完 9 题）实测 −0.11，落在跑间噪声 ±0.3 内；旧的自由读数模式是 **−0.765** |
| `iterate` | 在跑，**+1.02** | 19 张：29.03 → 30.05（引脚 F1 0.0088→0.0344、tp 引脚 1→18） |
| `rescan` / `know` / `conflict` | `rescan` 在跑；`know` / `conflict` **从未写出来** | — |

设计原则：**不推倒已有管线**。本层消费 `stage1` 的几何产物与 `stage2` 的
`_assigned/<case>.json`，输出一份改写过的 `assigned`，再交给
`stage2.assemble.build_doc` 重新装配。这样任何时候都可以只跑本层做对照。
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

__all__ = ["claims", "priors", "solve", "plausible", "llm_judge", "rescan",
           "iterate", "run"]
