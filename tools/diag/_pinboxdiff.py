# -*- coding: utf-8 -*-
"""逐 case 的 IC 引脚对账：GT vs 我方 `final_results/<case>_Pred.json`。

回答两件事：① 我们 emit 了哪些脚、GT 有哪些脚（键/名/点）；② 每根脚在**框外多少像素**
（GT 与我方各算一遍）—— 用来验证「GT 的引脚点约定是不是常数」。

★ 2026-10-09 用它查出的结论（`docs/RESULTS.md` §2.15）：**不是常数**。case 8 的
U1(PAW3395DM-T6QU) 左边 8 根 GT 点在框外 **22px**（画出来的引脚桩尖端，正好压在导线端点
x=972~974 上），右边 5 根在框外 **9px**（桩的中点；桩同样从框边画到框外 19px、导线接在
1192 的桩尖）。我们的点一律在框外 ~12px（`stage1/pins.py:55 PIN_OUT_BOX`）⇒ 左边 8 根必然
`position_mismatch`（10px > `pin_tol=5`），右边 5 根刚好过线。`docs/DIAGNOSTICS.md:760-781`
的 `_pintip.py` 给出总体 p50 = 13 / p75 = 15（逐脚不同的分布），所以「统一平移」「吸附到框边」
两条路（同节、`RESULTS.md:992`）判负是必然的。

坐标帧：GT 的 `*.json` 与我们的 `*_Pred.json` **都在 JSON 帧**（y 向上、原点左下），
`runs/stage1/<case>.stage1.json` 的 `bbox_img`/`pins` 在**图像帧**（y 向下）。别混。

用法：
    python -X utf8 eda-pcb\\tools\\diag\\_pinboxdiff.py --case 8 --run work\\_fast_out_dlv10
    python -X utf8 eda-pcb\\tools\\diag\\_pinboxdiff.py --case 8 --run work\\_fast_out_dlv10 --side
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # eda-pcb/tools/diag -> 仓库根
sys.path.insert(0, str(ROOT / "eda-pcb"))

GT_DIR = ROOT / "eda-pcb" / "yolo_part" / "Dataset(waitforpreprocess)" / "10GT"
IC_TYPES = ("box", "boxic", "ic")


def jload(p):
    with io.open(str(p), encoding="utf-8") as fp:
        return json.load(fp)


def out_dist(box, pt):
    """点到框的**带符号**外距：正 = 在框外，负 = 在框内（取四边里最近的那条）。"""
    x0, y0, x1, y1 = [float(v) for v in box]
    x, y = float(pt[0]), float(pt[1])
    return max(x0 - x, x - x1, y0 - y, y - y1)


def side_of(box, pt):
    x0, y0, x1, y1 = [float(v) for v in box]
    x, y = float(pt[0]), float(pt[1])
    return min([(abs(x - x0), "L"), (abs(x - x1), "R"),
                (abs(y - y0), "T"), (abs(y - y1), "B")])[1]


def centre(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="8")
    ap.add_argument("--run", default=r"work\_fast_out_dlv10",
                    help="产物目录（含 final_results/<case>_Pred.json）")
    ap.add_argument("--max", type=int, default=6, help="最多打印几颗 IC")
    ap.add_argument("--side", action="store_true", help="额外打印 GT 每侧框外距离的分位")
    a = ap.parse_args(argv)

    gt = jload(GT_DIR / a.case / (a.case + ".json"))
    gp = jload(ROOT / a.run / "final_results" / (a.case + "_Pred.json"))
    gt_comps = gt.get("components") or {}
    gt_pins = gt.get("pins") or {}
    pr_comps = gp.get("components") or {}
    pr_pins = gp.get("pins") or {}
    boxes = [(k, c) for k, c in gt_comps.items()
             if str(c.get("type", "")).lower() in IC_TYPES]
    pr_boxes = [(k, c) for k, c in pr_comps.items()
                if str(c.get("type", "")).lower() in IC_TYPES]
    print("GT 元件 %d（IC %d） 我方元件 %d（IC %d）" %
          (len(gt_comps), len(boxes), len(pr_comps), len(pr_boxes)))

    for gk, gc in boxes[:a.max]:
        gb = gc.get("bbox") or gc.get("box")
        best = None
        for pk, pc in pr_boxes:
            pb = pc.get("bbox") or pc.get("box")
            if not (gb and pb):
                continue
            d = math.hypot(centre(gb)[0] - centre(pb)[0], centre(gb)[1] - centre(pb)[1])
            if best is None or d < best[0]:
                best = (d, pk, pc)
        print("\n=== GT %s (%s) 框 %s" % (gk, gc.get("Name") or gc.get("name"),
                                          [round(float(v), 1) for v in (gb or [])]))
        if best:
            mb = best[2].get("bbox") or gb
            print("    我方 %s  d=%.1f  框 %s" % (best[1], best[0],
                                                  [round(float(v), 1) for v in mb]))
            for k, v in sorted((pr_pins.get(best[1]) or {}).items(), key=lambda kv: str(kv[0])):
                pt = v.get("point") or {}
                print("      me  %-10s name=%-9s (%.0f,%.0f)  框外 %5.1f px (%s)" %
                      (k, v.get("pinname"), pt.get("x", -1), pt.get("y", -1),
                       out_dist(mb, (pt.get("x", 0), pt.get("y", 0))),
                       side_of(mb, (pt.get("x", 0), pt.get("y", 0)))))
        for k, v in sorted((gt_pins.get(gk) or {}).items(), key=lambda kv: str(kv[0])):
            pt = v.get("point") or {}
            print("      GT  %-10s name=%-9s (%.0f,%.0f)  框外 %5.1f px (%s)" %
                  (k, v.get("pinname"), pt.get("x", -1), pt.get("y", -1),
                   out_dist(gb, (pt.get("x", 0), pt.get("y", 0))),
                   side_of(gb, (pt.get("x", 0), pt.get("y", 0)))))

    if a.side:
        print("\nGT 每侧框外距离：")
        agg = {}
        for gk, gc in boxes:
            gb = gc.get("bbox") or gc.get("box")
            if not gb:
                continue
            for _k, v in (gt_pins.get(gk) or {}).items():
                pt = v.get("point") or {}
                agg.setdefault(side_of(gb, (pt.get("x", 0), pt.get("y", 0))), []).append(
                    out_dist(gb, (pt.get("x", 0), pt.get("y", 0))))
        for s, vs in sorted(agg.items()):
            vs.sort()
            n = len(vs)
            print("   %s n=%-4d p10=%5.1f p50=%5.1f p90=%5.1f min=%5.1f max=%5.1f" %
                  (s, n, vs[int(0.1 * (n - 1))], vs[int(0.5 * (n - 1))], vs[int(0.9 * (n - 1))],
                   vs[0], vs[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
