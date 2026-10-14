# -*- coding: utf-8 -*-
"""§5.115：把我方 `stage1/pin_evidence.py` 与队友 PR#3 的「黑白框实测引脚」做**最小增量**合并。

只追加（不动既有函数）：
  _frame_outer / _one_pin_mono（他们的 _one_pin 变体）/ _black_ink / _text_mask /
  _normal_row / _frame_stroke_bands / recover_frame_pins

理由：直接整文件替换会让默认路径 84.95 -> 80.47（他们的 `_one_pin` 改成用外沿
`_frame_outer` 当探针锚点，而 `build_pin_evidence` 也吃这个改动）。所以保留我方
`_one_pin`，把他们的存成 `_one_pin_mono`，只由 `recover_frame_pins` 调用。

用法（cwd = eda-pcb）：python -X utf8 ..\work\_pinmono_patch.py
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OURS = ROOT / "eda-pcb" / "stage1" / "pin_evidence.py"
THEIRS = ROOT / "work" / "_wt_pr3" / "stage1" / "pin_evidence.py"

src_lines = THEIRS.read_text("utf-8").splitlines(keepends=True)


def seg(a: int, b: int) -> str:
    return "".join(src_lines[a - 1:b])


src = OURS.read_text("utf-8")
if "recover_frame_pins" in src:
    print("already patched")
    raise SystemExit(0)

header = (
    "\n\n"
    "# ---------------------------------------------------------------------------\n"
    "# §5.115 teammate PR#3 measured-monochrome-frame pin recovery: APPEND ONLY.\n"
    "# The existing build_pin_evidence / _one_pin / _candidate_edge are untouched:\n"
    "# their _one_pin probes from the outer frame edge (_frame_outer), and swapping\n"
    "# it in moves the default path 84.95 -> 80.47 (see §5.115). Theirs is kept as\n"
    "# _one_pin_mono and is called only by recover_frame_pins.\n"
    "# Knob: STAGE1_MONO_FRAME_PINS (default 0). Needs repo-root pinmarkers.py.\n"
    "# ---------------------------------------------------------------------------\n\n"
)

one_pin_mono = seg(221, 274).replace("def _one_pin(", "def _one_pin_mono(", 1)
tail = seg(310, 576).replace("_one_pin(", "_one_pin_mono(")
tail = tail.replace(
    "    if not math.isfinite(float(out_dist)) or float(out_dist) <= 0:",
    "    from pinmarkers import hollow_marker_bands\n"
    "    if not math.isfinite(float(out_dist)) or float(out_dist) <= 0:",
    1,
)

if "import copy" not in src:
    src = src.replace("import math\n", "import copy\nimport math\n", 1)

OURS.write_text(src + header + seg(97, 117) + "\n" + one_pin_mono + "\n" + tail, "utf-8")
print("patched; lines =", len(OURS.read_text("utf-8").splitlines()))
