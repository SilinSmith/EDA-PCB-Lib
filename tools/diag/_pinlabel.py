# -*- coding: utf-8 -*-
"""GT 的 IC 引脚名**在图上到底有没有被我们检出文字**，以及它落在元件框内还是框外。

动机（2026-10-08 实测）：
  * GT IC 引脚 `point` **100% 落在元件 bbox 之外**（到框边界 中位 9px / p90 35px / p99 67px）；
  * 我们产出的引脚里 **78.2% 的 `pinname` 就是引脚号本身**（`assemble.py:761 nm = r["nm"] or num`）；
  * 而 `stage2/assign.py:969` 对 `pin_name` 角色有一条 **`if want_inside and not t["inside"]: continue`**
    —— 只收**框内**的文字当引脚名。

若 GT 的引脚名标签本来就画在框外，那这条 `want_inside` 就是「78% 引脚没名字」的直接原因。

用法：
    python work\\tools\\diag\\_pinlabel.py --n 60
    python work\\tools\\diag\\_pinlabel.py --n 5 --each
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import IC_TYPES, canon_type, normalise_doc, _point  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="200_train_cases")
    ap.add_argument("--out", default=str(ROOT / "work" / "_submit_out220r36"))
    ap.add_argument("--pad", type=float, default=6.0, help="点落在文本框外多少 px 内算命中")
    ap.add_argument("--each", action="store_true")
    a = ap.parse_args()

    st = Counter()
    ex = []
    cs = [c for c in discover(DEFAULT_DATA_ROOT) if c.split == a.split][:a.n]
    for c in cs:
        sp = Path(a.out) / "runs" / "stage1" / (c.case_id + ".stage1.json")
        if not sp.exists():
            continue
        s1 = load_json(sp)
        H = float((s1.get("image") or {}).get("h") or 0)
        if H <= 0:
            continue
        # 文本框统一到 GT 的坐标系（y 向上）：bbox_img=[x1,y1,x2,y2]（y 向下）
        tbs = []
        for t in s1.get("texts") or []:
            b = t.get("bbox_img")
            if not (isinstance(b, (list, tuple)) and len(b) >= 4):
                continue
            x1, y1, x2, y2 = [float(v) for v in b[:4]]
            tbs.append((min(x1, x2), H - max(y1, y2), max(x1, x2), H - min(y1, y2)))
        gt = normalise_doc(load_json(c.json_path))
        for comp, v in (gt.get("components") or {}).items():
            if canon_type((v or {}).get("type")) not in IC_TYPES:
                continue
            bb = v.get("bbox")
            if isinstance(bb, dict):
                bx = [bb.get("x1"), bb.get("y1"), bb.get("x2"), bb.get("y2")]
            elif isinstance(bb, (list, tuple)) and len(bb) >= 4:
                bx = list(bb[:4])
            else:
                bx = None
            if bx and all(t is not None for t in bx):
                x1, y1, x2, y2 = [float(t) for t in bx]
                cbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            else:
                cbox = None
            for pk, g in ((gt.get("pins") or {}).get(comp) or {}).items():
                if not isinstance(g, dict):
                    continue
                p = _point(g.get("point"))
                if not p:
                    st["pins"] += 1
                    st["no_point"] += 1
                    continue
                st["pins"] += 1
                px, py = p
                best, bd = None, 1e9
                for (tx1, ty1, tx2, ty2) in tbs:
                    dx = max(tx1 - px, px - tx2, 0.0)
                    dy = max(ty1 - py, py - ty2, 0.0)
                    d = math.hypot(dx, dy)
                    if d < bd:
                        bd, best = d, (tx1, ty1, tx2, ty2)
                if best is None or bd > a.pad:
                    st["no_text"] += 1
                    if a.each and len(ex) < 20:
                        ex.append((c.case_id, comp, pk, str(g.get("pinname")),
                                   round(bd, 1), "无文字", None))
                    continue
                st["text"] += 1
                cx, cy = (best[0] + best[2]) / 2.0, (best[1] + best[3]) / 2.0
                inside = None
                if cbox:
                    inside = (cbox[0] <= cx <= cbox[2] and cbox[1] <= cy <= cbox[3])
                if inside is True:
                    st["text_inside"] += 1
                elif inside is False:
                    st["text_outside"] += 1
                else:
                    st["text_nobox"] += 1
                if a.each and inside is False and len(ex) < 20:
                    ex.append((c.case_id, comp, pk, str(g.get("pinname")),
                               round(bd, 1), "框外", [round(t) for t in best]))
    print("case %d 张，GT IC 引脚 %d 根" % (len(cs), st["pins"]))
    print("  该点附近 %.0fpx 内**没有任何检出文本框**：%d（%.1f%%）" % (
        a.pad, st["no_text"], 100.0 * st["no_text"] / max(1, st["pins"])))
    print("  有文本框：%d（%.1f%%）" % (st["text"], 100.0 * st["text"] / max(1, st["pins"])))
    print("      其中文本中心在**元件框内**：%d（%.1f%% of 有文字）" % (
        st["text_inside"], 100.0 * st["text_inside"] / max(1, st["text"])))
    print("      其中文本中心在**元件框外**：%d（%.1f%% of 有文字）" % (
        st["text_outside"], 100.0 * st["text_outside"] / max(1, st["text"])))
    print("      元件无 bbox：%d" % st["text_nobox"])
    for e in ex:
        print("  ", e)


if __name__ == "__main__":
    raise SystemExit(main())
