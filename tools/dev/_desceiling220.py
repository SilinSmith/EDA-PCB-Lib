# -*- coding: utf-8 -*-
"""把 `eda-pcb/tools/diag/_des_ceiling.py` 指到 220 张的 stage1 产物上跑（原工具写死 work\\stage1_all14）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SRC = Path(r"Y:\workspace\EDA\eda-pcb\tools\diag\_des_ceiling.py")
spec = importlib.util.spec_from_file_location("dsc", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)                       # type: ignore[union-attr]
mod.S1 = Path(r"Y:\workspace\EDA\work\_full220\runs\stage1")
argv = sys.argv[1:] or ["--limit", "6", "--split", "200_train_cases"]
raise SystemExit(mod.main(argv))
