# -*- coding: utf-8 -*-
"""stage2 —— 第二阶段：文字识别与结构组装。

    from stage2.run import run_cases
    run_cases(stage1_dir, out_dir, limit=5)

阶段划分（与用户的口径一致）：
  stage1  几何：元件框、导线、引脚点           -> `<case>.stage1.json`
  stage2  本地：文字框 + CRNN 识别 + 角色分配   -> `<case>.json`（官方格式）
  stage3  LLM：对低置信度/自相矛盾的槽位做多选仲裁
"""
from .text import TextEngine, detect_texts, imread_unicode, read_stage1_texts
from .assign import assign
from .assemble import build_doc

__all__ = ["TextEngine", "detect_texts", "imread_unicode", "read_stage1_texts",
           "assign", "build_doc"]
