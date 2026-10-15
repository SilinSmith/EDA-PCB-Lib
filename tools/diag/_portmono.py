# -*- coding: utf-8 -*-
"""§5.114 移植：把队友 PR#3 的**实测黑白框引脚校准**搬进我方 stage1。

只做机械搬运，不改语义：
  1. `work/_wt_pr3/pinmarkers.py`            -> `eda-pcb/pinmarkers.py`（新文件）
  2. `work/_wt_pr3/stage1/pin_evidence.py`   -> `eda-pcb/stage1/pin_evidence.py`（超集覆盖）
  3. `eda-pcb/stage1/pipeline.py`：插入旋钮 `MONO_FRAME_PINS_ON`（默认 0）
     以及队友的 `apply_mono_frame_pins` / `overlay_mono_evidence` 两个函数原文
     （从队友文件里按 AST 行号切出来，保证逐字一致）。

调用点（我们的变量名、放在哪儿）随后手工插入 —— 见脚本末尾提示。
用法：python -X utf8 work\\_portmono.py
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

THEIRS = Path(r"Y:\workspace\EDA\work\_wt_pr3")
OURS = Path(r"Y:\workspace\EDA\eda-pcb")
FUNCS = ("apply_mono_frame_pins", "overlay_mono_evidence")

# ---- 1/2 文件级搬运 -------------------------------------------------------
copy_pairs = [("pinmarkers.py", "pinmarkers.py"),
              ("stage1/pin_evidence.py", "stage1/pin_evidence.py")]
for src_rel, dst_rel in copy_pairs:
    src, dst = THEIRS / src_rel, OURS / dst_rel
    if not src.exists():
        raise SystemExit("队友文件不存在: %s" % src)
    shutil.copyfile(src, dst)
    print("搬运 %-28s -> %-28s (%d 行)" % (src_rel, dst_rel,
          len(dst.read_text(encoding="utf-8").splitlines())))

# ---- 3 pipeline.py 插入 ---------------------------------------------------
their_text = (THEIRS / "stage1/pipeline.py").read_text(encoding="utf-8")
tree = ast.parse(their_text)
their_lines = their_text.splitlines(keepends=True)
blocks = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
        blocks[node.name] = "".join(their_lines[node.lineno - 1:node.end_lineno])
missing = [f for f in FUNCS if f not in blocks]
if missing:
    raise SystemExit("队友文件里找不到函数: %s" % missing)

our_path = OURS / "stage1/pipeline.py"
our_text = our_path.read_text(encoding="utf-8")
if "apply_mono_frame_pins" in our_text:
    raise SystemExit("我方 pipeline.py 里已经存在 apply_mono_frame_pins —— 先撤销再跑")
if "MONO_FRAME_PINS_ON" not in our_text:
    anchor = [ln for ln in our_text.splitlines(keepends=True)
              if ln.startswith("PIN_COLOR_ON = env_on(")]
    if not anchor:
        raise SystemExit("找不到锚点 PIN_COLOR_ON = env_on(")
    knob = (
        "# ★★ 黑白框实测引脚（队友 PR#3：`stage1/pin_evidence.py` 新增的\n"
        "#   `recover_frame_pins`/`_frame_outer`/`_black_ink`）—— 把「按几何/文字猜出来的」\n"
        "#   引脚点校准到**实测的框边笔画行**上。它只校准点、不增删引脚，且内部有三道\n"
        "#   ValueError 守卫：不许改元件索引集合、不许改旧引脚索引、不许动已实测的彩色触点。\n"
        "#   默认关：先 A/B（§5.114）；配 `STAGE1_MONO_FRAME_PINS=1` 打开。\n"
        "MONO_FRAME_PINS_ON = env_on(\"STAGE1_MONO_FRAME_PINS\", \"0\")\n")
    our_text = our_text.replace(anchor[0], anchor[0] + knob, 1)
    print("插入旋钮 MONO_FRAME_PINS_ON（默认 0）")

our_tree = ast.parse(our_text)
anchor_line = None
for node in our_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "overlay_color_evidence":
        anchor_line = node.end_lineno
if anchor_line is None:
    raise SystemExit("找不到我方 overlay_color_evidence 函数")
our_lines = our_text.splitlines(keepends=True)
header = ("\n\n# ===== 以下两函数逐字取自队友 `codex/combined-pin-recognition`（PR#3）\n"
          "#       `stage1/pipeline.py`；我们这边只把它们接在引脚定稿之后、build_nets 之前。\n")
inserted = header + "\n\n" + blocks["apply_mono_frame_pins"] + "\n\n" + blocks["overlay_mono_evidence"]
our_lines.insert(anchor_line, inserted)
our_path.write_text("".join(our_lines), encoding="utf-8")
print("插入 apply_mono_frame_pins(%d 行) + overlay_mono_evidence(%d 行) 到 pipeline.py 第 %d 行之后"
      % (blocks["apply_mono_frame_pins"].count("\n"),
         blocks["overlay_mono_evidence"].count("\n"), anchor_line))
print("\n还需要手工做的调用点：")
print("  (a) in the PIN_COLOR_ON block region:  pin_mono = {}  /  pin_connected = {}  defaults")
print("  (b) after that block, before build_nets:")
print("        if MONO_FRAME_PINS_ON:")
print("            try:")
print("                comp_pins, pin_synth, color_overlays, pin_connected, pin_mono = \\")
print("                    apply_mono_frame_pins(comps, comp_pins, bgr, _ink, texts,")
print("                                         pin_synth, color_overlays, pin_connected)")
print("            except Exception as e: ... errs.append(f\"pin_mono: ...\")")
print("  (c) replacement of  pin_evidence = overlay_color_evidence(...)  by the")
print("      mono-then-color order:  overlay_mono_evidence(pin_evidence, pin_mono, comp_pins)")
