"""引脚取景「旧(cross 对称) vs 新(按引脚间距收窄垂直半宽)」对照图 —— 纯 CPU，不叫 LLM。

用法：
    python work\\tools\\diag\\_pincropab.py <case> [元件键]
默认 case 8 / 自动挑引脚最多的那个元件（通常是那颗大 IC）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D                          # noqa: E402
from stage2.text import imread_unicode                          # noqa: E402
from vlm import enrich as E                                     # noqa: E402

PRED_DIR = Path(r"Y:\workspace\EDA\work\_fusion_out220no4")
OUT = Path(r"Y:\workspace\EDA\work\preview")

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
outp = Path(r"Y:\workspace\EDA\work\preview\_pincrop_new.png")
for a in sys.argv[1:]:
    if a.startswith("--out="):
        outp = Path(a.split("=", 1)[1])

CASE = argv[0] if argv else "8"
KEY = argv[1] if len(argv) > 1 else None

case = D.find_case(CASE)
pred = json.loads((PRED_DIR / f"{case.case_id}.json").read_text(encoding="utf-8"))
bgr = imread_unicode(case.png_path)

comps = pred.get("components") or {}
pins = pred.get("pins") or {}
if KEY is None:
    KEY = max(pins, key=lambda k: len(pins[k] or {}))
bb = comps[KEY]["bbox"]
pt_list = [(v["point"]["x"], v["point"]["y"]) for v in (pins[KEY] or {}).values()]
ctr = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)

opts = {"inner": 78.0, "cross": 58.0, "outer": 34.0, "ref": 762.0,
        "mr": 4, "mt": 2, "color": "red" if "red" in E.COLORS else list(E.COLORS)[0]}
win = E.adaptive_window(bb, opts["inner"], opts["cross"], opts["outer"], opts["ref"])
pitch = E._pin_pitch(pt_list)
cp = min(win[2], max(E.DEFAULTS["pin_cross_min"], E.DEFAULTS["pin_pitch_frac"] * pitch)) \
    if pitch > 0 else win[2]

print(f"case {case.case_id}  元件 {KEY}  n_pins={len(pt_list)}")
print(f"  bbox  {[round(float(v), 1) for v in bb]}  "
      f"尺寸 {abs(bb[2]-bb[0]):.0f}x{abs(bb[3]-bb[1]):.0f}")
print(f"  win(outer,inner,cross) = ({win[0]:.1f}, {win[1]:.1f}, {win[2]:.1f})")
print(f"  引脚间距 pitch = {pitch:.1f}px  垂直半宽 cross_perp = {cp:.1f}px   "
      f"⇒ 窗口垂直总高 {2*cp:.0f}px（旧 {2*win[2]:.0f}px）")

old_c, new_c = [], []
for q in pt_list:
    a = E.crop_dir(bgr, q, ctr, win[0], win[1], win[2])
    b = E.crop_dir(bgr, q, ctr, win[0], win[1], win[2], cross_perp=cp,
                   dir_uv=E._edge_dir((bb[0], bb[1], bb[2], bb[3]), q))
    if a is None or b is None:
        continue
    ia, ra = a
    ib, rb = b
    E.mark(ia, ra, opts["mr"], opts["mt"], E.COLORS[opts["color"]])
    E.mark(ib, rb, opts["mr"], opts["mt"], E.COLORS[opts["color"]])
    old_c.append(ia)
    new_c.append(ib)

cols = 6
g_old = E.tile(old_c[:cols * 3], cols=cols)
g_new = E.tile(new_c[:cols * 3], cols=cols)
sep = np.full((14, g_old.shape[1], 3), 0, np.uint8)
cv2.putText(g_old, "OLD cross=%.0f (h=%.0f)" % (win[2], 2 * win[2]), (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)
cv2.putText(g_new, "NEW pitch=%.0f cross_perp=%.0f (h=%.0f)" % (pitch, cp, 2 * cp), (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2, cv2.LINE_AA)
cat = np.vstack([g_new, sep, g_old])
outp.parent.mkdir(parents=True, exist_ok=True)
cv2.imencode(".png", cat)[1].tofile(str(outp))
print("写出:", outp)
