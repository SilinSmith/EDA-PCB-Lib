# -*- coding: utf-8 -*-
"""验证「假值词表统一」：把每个手写短词表换成 `envflag.env_on` 之后，
`off` / `none` / `disable` 这些词是否真的能**关掉**功能。

用法（当前 shell 的环境变量生效）：
    python work\\tools\\diag\\_envflagprobe.py
"""
import os
import sys

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")

import fusion.priors as P          # noqa: E402
import fusion.llm_judge as J       # noqa: E402
import fusion.audit as A           # noqa: E402
import fusion.run as R             # noqa: E402
import fusion.solve as S           # noqa: E402
import fusion.cropfix as C         # noqa: E402
import importlib                 # noqa: E402
# 注意：`stage2/__init__.py` 里 `from .assign import assign` 会把 `<pkg>.assign`
# 这个**子模块属性**覆盖成同名函数，于是 `import stage2.assign as SA` 拿到的是
# **函数**（`AttributeError: 'function' object has no attribute 'PIN_ALONG'`）。
# 必须走 importlib 拿模块本体。
SA = importlib.import_module("stage2.assign")   # noqa: E402

rows = [
    ("FUSION_SHAPE_GATE", "off", P.SHAPE_GATE),
    ("FUSION_JUDGE", "none", J.JUDGE),
    ("FUSION_JUDGE_OTHER", "disable", J.REJECT_OTHER),
    ("FUSION_AUDIT_REREAD", "no", A.REREAD),
    ("FUSION_AUDIT", "0", A.ON),
    ("FUSION_LLM", "off", R.DO_LLM),
    ("FUSION_GAP_FILL", "none", S.GAP_FILL),
    ("FUSION_CROPFIX", "OFF", C.ON),
    ("", "none", SA.PIN_ALONG),
]
for name, val, got in rows:
    env = os.environ.get(name)
    ok = "OK " if got is False else "!! "
    print("%s%-22s = %-8r (env=%r) -> %s" % (ok, name, val, env, got))
