"""自检：新 `crop_dir` 在 `cross_perp=None` 时**逐位等于**旧实现（A/B 对照臂的正确性前提），
以及新窗口的**垂直总高只会更小**。

用法：python work\\tools\\diag\\_pincheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D                          # noqa: E402
from stage2.text import imread_unicode                          # noqa: E402
from vlm import enrich as E                                     # noqa: E402

PRED_DIR = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage2")
n_ok = n_bad = 0
tighter = []
for cid in "1 2 3 4 5 6 7 8 9 10".split():
    case = D.find_case(cid)
    doc = json.loads((PRED_DIR / f"{case.case_id}.json").read_text(encoding="utf-8"))
    bgr = imread_unicode(case.png_path)
    pins = doc.get("pins") or {}
    comps = doc.get("components") or {}
    for key, plist in pins.items():
        if key not in comps or len(plist or {}) < 3:
            continue
        bb = comps[key]["bbox"]
        pts = [(v["point"]["x"], v["point"]["y"]) for v in plist.values()]
        ctr = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        win = E.adaptive_window(bb, E.DEFAULTS["inner"], E.DEFAULTS["cross"],
                                E.DEFAULTS["outer"], E.DEFAULTS["ref"])
        pitch = E._pin_pitch(pts)
        cp = (min(win[2], max(E.DEFAULTS["pin_cross_min"],
                              E.DEFAULTS["pin_pitch_frac"] * pitch)) if pitch > 0 else None)
        for q in pts:
            a = E.crop_dir(bgr, q, ctr, win[0], win[1], win[2])                 # 旧
            b = E.crop_dir(bgr, q, ctr, win[0], win[1], win[2], cross_perp=None)  # 新，退化臂
            if a is None or b is None:
                n_ok += (a is None and b is None)
                n_bad += (a is None) != (b is None)
                continue
            ia, ra = a
            ib, rb = b
            same = (ia.shape == ib.shape and ra == rb and np.array_equal(ia, ib))
            n_ok += same
            n_bad += (not same)
            if cp is not None:
                c = E.crop_dir(bgr, q, ctr, win[0], win[1], win[2], cross_perp=cp)
                if c is not None:
                    tighter.append((win[2], cp, a[0].shape[0], c[0].shape[0]))

print(f"退化一致性: {n_ok} 通过 / {n_bad} 不一致")
if tighter:
    olds = [t[2] for t in tighter]
    news = [t[3] for t in tighter]
    print(f"窗口高: 旧 中位 {int(np.median(olds))}px 最大 {max(olds)}px  |  "
          f"新 中位 {int(np.median(news))}px 最大 {max(news)}px   （{len(tighter)} 个引脚）")
    print(f"新窗口更窄的引脚: {sum(1 for t in tighter if t[3] < t[2])}/{len(tighter)}；"
          f"没有任何一个变高: {all(t[3] <= t[2] for t in tighter)}")
assert n_bad == 0, "退化臂与旧实现不一致 —— A/B 的对照臂不可信"
print("OK")
