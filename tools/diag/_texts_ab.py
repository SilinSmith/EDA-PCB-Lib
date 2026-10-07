# 文字框来源 A/B（在**当前**管线上重跑）：--texts {stage1,detect,union,auto}
# 早前（分数 ~6.7 时代）结论是 stage1 最好；管线已大改（PIN_STAMP / value 门控），重测。
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA\eda-pcb")
S1 = r"Y:\workspace\EDA\work\stage1_v6"
PY = r"K:\python\311\python.exe"
W = Path(r"Y:\workspace\EDA\work")

for mode in ("stage1", "union", "detect", "auto"):
    out = W / ("_ab_texts_" + mode)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, "-m", "stage2.run", "--stage1", S1, "--out", str(out),
                        "--texts", mode], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    e = subprocess.run([PY, str(W / "tools" / "eval" / "stage2_eval.py"), str(out)],
                       cwd=str(ROOT), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    print("=== --texts %s ===" % mode, flush=True)
    for ln in (e.stdout or "").splitlines():
        if ln.startswith("  ") or "平均权重得分" in ln or "失配明细" in ln:
            print(ln.rstrip(), flush=True)
