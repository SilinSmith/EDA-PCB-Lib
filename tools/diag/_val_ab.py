# 值门控 A/B：STAGE2_VALUE_UNIT × STAGE2_VALUE_STEAL
# 用法：python work\tools\diag\_val_ab.py
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA\eda-pcb")
S1 = r"Y:\workspace\EDA\work\stage1_v6"
PY = r"K:\python\311\python.exe"

ARMS = [
    ("v9_base", "0", "0"),
    ("unit", "1", "0"),
    ("steal40", "0", "40"),
    ("unit_steal40", "1", "40"),
    ("unit_steal80", "1", "80"),
]

for name, unit, steal in ARMS:
    out = Path(r"Y:\workspace\EDA\work") / ("_ab_" + name)
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               STAGE2_VALUE_UNIT=unit, STAGE2_VALUE_STEAL=steal)
    r = subprocess.run([PY, "-m", "stage2.run", "--stage1", S1, "--out", str(out)],
                       cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    e = subprocess.run([PY, str(ROOT / ".." / "work" / "tools" / "eval" / "stage2_eval.py"),
                        str(out)], cwd=str(ROOT), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    line = ""
    for ln in (e.stdout or "").splitlines():
        if "平均" in ln or "得分" in ln or "component" in ln.lower() or "F1" in ln:
            line += ln.strip() + " | "
    print("%-14s UNIT=%s STEAL=%-3s %s" % (name, unit, steal, tail[0]), flush=True)
    print("    ", (e.stdout or "").strip()[-600:], flush=True)
