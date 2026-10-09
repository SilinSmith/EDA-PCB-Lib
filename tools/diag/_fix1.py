# -*- coding: utf-8 -*-
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "work" / "tools" / "diag"))
from _subst import subst  # noqa: E402

P = ROOT / "work" / "flow" / "_wfbullets.py"

pairs = [
    ('"llm_judge": "llm_judge.py：只做候选二选一，安全的中性；真正限制是问不到该问的"',
     '"llm_judge": "llm_judge.py：候选二选一 + **上下文块（官方 9 张 +0.79）**；整表判负"'),
]

h, t, s = subst(P, pairs)
print("命中 %d/%d" % (h, t))
