"""跑一次融合层并把它内部的 log 全打出来（看 LLM 轮到底做了什么）。

用法：
    python work\\tools\\diag\\_judgeprobe.py <case> <run_dir>
其中 run_dir 是 `submission\\run.py -output` 那一级（里面要有 runs\\stage1 等）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))

case = sys.argv[1] if len(sys.argv) > 1 else "0143"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "work" / "_pin1_sheetM"
runs = out / "runs"

os.environ.setdefault("FUSION_RUN_DIR", str(runs))
os.environ.setdefault("FUSION_OUT_DIR", str(runs / "fusion"))
os.environ.setdefault("FUSION_VLM_DIR", str(runs / "stage3_vlm"))
os.environ.setdefault("FUSION_BASE", "ours")
os.environ.setdefault("FUSION_ITERATE", "1")

from fusion import run as fr          # noqa: E402
from fusion import llm_judge as LJ    # noqa: E402
from fusion import claims as CL       # noqa: E402

print("== 配置 ==")
print("  DO_LLM      :", fr.DO_LLM)
print("  MAX_ROUNDS  :", fr.MAX_ROUNDS)
print("  JUDGE       :", LJ.JUDGE)
print("  MODE        :", LJ.MODE)
print("  SHEET_ON    :", LJ.SHEET_ON)
print("  MAX_ASK     :", LJ.MAX_ASK)
print("  LLM_SCORE   :", LJ.LLM_SCORE)

rep = fr.run_case(case, log=print)
print("\n== run_case 返回 ==")
for k, v in rep.items():
    if isinstance(v, (dict, list)):
        print("  %-16s %s" % (k, str(v)[:400]))
    else:
        print("  %-16s %s" % (k, v))
