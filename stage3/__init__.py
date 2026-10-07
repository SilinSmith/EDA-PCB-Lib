# -*- coding: utf-8 -*-
"""第三阶段：把第二阶段的歧义槽位交给 LLM（或本地 stub）做选择题。

对外只暴露三件事：
    build(case_id, assigned, components)  -> CaseHypotheses
    arbitrate(hy, mode="auto", ...)       -> 统计
    apply(assigned, hy)                   -> 把选定值写回 assigned（供重装配）
以及 CLI `python -m stage3.run`。
"""
from .arbitrate import DESIG_SLOT, apply, arbitrate, build          # noqa: F401

__all__ = ["build", "arbitrate", "apply", "DESIG_SLOT"]
