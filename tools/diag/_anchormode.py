# -*- coding: utf-8 -*-
"""锚点**口径判别器**：新 GT 一到就直接告诉我们「GT 的脚锚点钉在哪个几何参照上」。

这是**量尺侧**工具（故意用 GT 锚点做被测量）：对手上每根可配对的脚，沿该脚的外向轴
取四个候选参照，再看 GT 锚点相对它们各自差多少：

    我方锚点   —— 我们现在交付的点（残差 = 沿轴投影距离，非零才说明口径不同）
    引线外端   —— 沿外向轴最外侧的 maroon 像素（tip，§5.152 补里 t 分布很散）
    交点       —— 绿像素且内侧贴 maroon（junc，`_juncprobe` 同规则）
    首绿       —— 沿轴最近的一个绿像素（green，不看 maroon）

判据（写死，免得事后解释）：**可用** = 该参照对这批脚有定义的比例 ≥ 60%；
**采纳** = 在可用参照里，残差 |中位| 最小且 p90-p10 最紧；若最紧的也 > 4px，
就明确说「本批 GT 的锚点不落在任何一个候选参照上，需要人类说明口径」。

用法：
  python -X utf8 eda-pcb/tools/diag/_anchormode.py
  python -X utf8 eda-pcb/tools/diag/_anchormode.py --min-n 3 --show
  python -X utf8 eda-pcb/tools/diag/_anchormode.py --run-dir work/_fast_out_delivery_default
"""
import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, r"Y:\workspace\EDA\eda-pcb")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schematic_parser.data import discover                                     # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components, rewrite_keys  # noqa: E402
import _pinanchor as PA                                                         # noqa: E402
import _juncprobe as JP                                                         # noqa: E402

RUN = r"Y:\workspace\EDA\work\_fast_out_deterministic_default"
REFS = ("ours", "tip", "junc", "green")


def _proj(q, g, u):
    """GT 锚点相对我方锚点的：沿外向轴的投影距离、以及垂直轴的偏离。"""
    dx, dy = g[0] - q[0], g[1] - q[1]
    along = dx * u[0] + dy * u[1]
    perp = abs(dx * u[1] - dy * u[0])
    return along, perp


def _tip(img, q, u, half):
    """沿外向轴最外侧的 maroon 像素 t（找不到/顶到上限都返回原始信息）。"""
    h, w = img.shape[0], img.shape[1]
    last = None
    for t in range(-half, half + 1):
        x, y = int(round(q[0] + u[0] * t)), int(round(q[1] + u[1] * t))
        if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
            continue
        if PA._classify(img, x, y) == "m":
            last = t
    return last


def scan(a, cfg, index):
    acc = {r: dict(res=[], perp=[], undef=0, sat=0, n=0) for r in REFS}
    group_dev = {r: [0, 0] for r in REFS}          # [恒定组数, n>=3 组数]
    rows = []
    for cid in a.cases:
        case = index.get(str(cid))
        if case is None:
            continue
        pp = os.path.join(a.run_dir, "final_results", "%s_Pred.json" % cid)
        if not os.path.exists(pp):
            continue
        gt_doc = json.load(io.open(case.json_path, encoding="utf-8"))
        pred_doc = json.load(io.open(pp, encoding="utf-8"))
        _, km = assign_components(gt_doc, pred_doc, cfg, {})
        pred2 = rewrite_keys(pred_doc, km)
        r = PA.analyse_case(str(cid), case, a.run_dir, cfg, ray=False)
        if r is None:
            continue
        img = np.array(Image.open(case.png_path).convert("RGB"))
        h = img.shape[0]
        n_pin = 0
        res_by_group = {rf: {} for rf in REFS}
        for (ck, edge), g in r["groups"].items():
            u = PA._outward_img(edge)
            for num, pt, gpt, _d in g["td"]:
                n_pin += 1
                q = (pt[0], h - pt[1])
                gt_q = (gpt[0], h - gpt[1])
                along, perp = _proj(q, gt_q, u)
                t_tip = _tip(img, q, u, a.half)
                p_j = JP.probe(img, q, u, a.half, True, lat=a.lat, back=a.back)
                p_g = JP.probe(img, q, u, a.half, False, lat=a.lat, back=a.back)
                cand = {"ours": 0.0, "tip": t_tip,
                        "junc": None if p_j is None else p_j[0],
                        "green": None if p_g is None else p_g[0]}
                for rf in REFS:
                    c = cand[rf]
                    if c is None:
                        acc[rf]["undef"] += 1
                        continue
                    acc[rf]["n"] += 1
                    acc[rf]["res"].append(along - c)
                    acc[rf]["perp"].append(perp)
                    res_by_group[rf].setdefault((ck, edge), []).append(along - c)
                if t_tip is not None and abs(t_tip) >= a.half - 1:
                    acc["tip"]["sat"] += 1
                if a.show:
                    rows.append((str(cid), ck, num, edge, along, perp, t_tip,
                                 None if p_j is None else p_j[0],
                                 None if p_g is None else p_g[0]))
        for rf in REFS:
            for _k, v in res_by_group[rf].items():
                if len(v) >= max(3, a.min_n):
                    group_dev[rf][1] += 1
                    if PA._axis_dev([(0.0, float(x)) for x in v]) <= 2.0:
                        group_dev[rf][0] += 1
        print("case %-3s 可配对脚 %3d" % (cid, n_pin))
    return acc, group_dev, rows


def report(acc, group_dev, n_pin, a):
    print()
    print("脚总数 %d；口径：half=%d, lat=±%d, back=1..%d, run-dir=%s"
          % (n_pin, a.half, a.lat, a.back, a.run_dir))
    print("%-6s %-14s %-9s %-18s %-14s %-16s %s"
          % ("参照", "有定义", "残差中位", "p10 / p90", "|残差|<=2px", "逐边恒定(n>=3)",
             "备注"))
    order = []
    for rf in REFS:
        d = acc[rf]
        res = np.array(d["res"]) if d["res"] else np.array([np.nan])
        med = float(np.nanmedian(res))
        p10, p90 = float(np.nanpercentile(res, 10)), float(np.nanpercentile(res, 90))
        share = float(np.mean(np.abs(res) <= 2.0))
        gs, gn = group_dev[rf]
        note = []
        if rf == "tip" and d["sat"]:
            note.append("顶到上限 %d" % d["sat"])
        if d["undef"]:
            note.append("无定义 %d" % d["undef"])
        print("%-6s %5d (%5.1f%%)   %+7.2f   (%+6.2f,%+6.2f)   %5.1f%%          %2d/%2d (%5.1f%%)   %s"
              % (rf, d["n"], 100.0 * d["n"] / max(1, n_pin), med, p10, p90,
                 100.0 * share, gs, gn, 100.0 * gs / max(1, gn), " ".join(note)))
        order.append((rf, d["n"] / max(1, n_pin), abs(med), p90 - p10, med))
    usable = [x for x in order if x[1] >= 0.60 and x[0] != "ours"]
    tight = min([x for x in order if x[0] != "ours"], key=lambda x: x[3])
    print("参考：所有候选里跨度最小的是 %s（可用率 %.1f%%、残差中位 %+.2f、跨度 %.2f px，"
          "但可能大量无定义/顶上限）。" % (tight[0], 100.0 * tight[1], tight[4], tight[3]))
    if not usable:
        print("判别：没有任何候选参照达到 60% 可用率 ⇒ 得先修检测器（或人类给口径）。")
        return
    best = min(usable, key=lambda x: (x[3], x[2]))
    print("判别：可用率 >=60%% 的候选里最紧的是 **%s**（可用率 %.1f%%、残差中位 %+.2f、跨度 %.2f px）。"
          % (best[0], 100.0 * best[1], best[4], best[3]))
    if tight[3] <= 4.0:
        print("      ⇒ 本批 GT 的锚点基本落在 **%s** 上（可用率 %.1f%%、跨度 %.2f <= 4px）：实装方式 = "
              "把锚点搬到该参照，再按「净增益>0 且两口径对照不降」A/B 定去留。"
              % (tight[0], 100.0 * tight[1], tight[3]))
    else:
        print("      ⇒ 所有候选参照里跨度最小的 %s 也有 %.2f px（> 4px）：本批 GT 的锚点**不落在**任何"
              "候选参照上，需要人类说明口径，别硬凑。" % (tight[0], tight[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--cases", default=",".join(str(i) for i in range(1, 11)))
    ap.add_argument("--half", type=int, default=26)
    ap.add_argument("--lat", type=int, default=2)
    ap.add_argument("--back", type=int, default=2)
    ap.add_argument("--min-n", type=int, default=2, help="每组至少几根脚才参与「逐边恒定」")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    a.cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    cfg = EvalConfig()
    index = {str(c.case_id): c for c in discover()}
    acc, group_dev, rows = scan(a, cfg, index)
    n_pin = max(acc["ours"]["n"] + acc["ours"]["undef"], 1)
    if a.show:
        print()
        print("%-5s %-6s %-4s %-3s %8s %7s %7s %7s %7s"
              % ("case", "comp", "pin", "边", "GT投影", "垂轴", "tip", "junc", "green"))
        for r in rows:
            print("%-5s %-6s %-4s %-3s %+8.2f %7.2f %7s %7s %7s"
                  % (r[0], r[1], r[2], r[3], r[4], r[5],
                     "-" if r[6] is None else "%d" % r[6],
                     "-" if r[7] is None else "%d" % r[7],
                     "-" if r[8] is None else "%d" % r[8]))
    report(acc, group_dev, n_pin, a)


if __name__ == "__main__":
    main()
