# -*- coding: utf-8 -*-
"""位号缺口的**最上游**归因：那个文字框**到底被检出来没有**。

`_desigassoc.py` 已判死「关联错」这条路：没配上的 629 颗 GT IC 里
  (i)  正确位号在其自己的候选表里      18 颗 / 310 引脚（6.0%）
  (ii) 正确位号在**别的** cid 的表里    24 颗 / 141 引脚（2.7%），**纯交换 0 颗**
  (iii) 全表都没有                   586 颗 / 4756 引脚（**91.3%**）
⇒ 融合层的位号「选择/重归属」最多只值 6%，九成以上是**这个字符串根本没进池子**。

本脚本回答「为什么没进池子」，只有两种可能，修法完全不同：

  A. **检测就没检到** —— 这张图上，GT 那颗 IC 的邻域里**一个 stage1 文字框都没有**。
     要动检测/文字分割（`stage1` 的 texts 生成）。
  B. **检到了但没归属/读错** —— 邻域**有**文字框，只是没被认领成这颗元件的位号
     （或认领了但字符串错）。要动归属与识别（`stage2` 的 assign/TextEngine）。

判据：把 GT 元件框按 `pad` 外扩，数 `stage1.texts[].bbox_img` 里
有多少框与它相交（`_assigned` 只 dump 了被认领的框，**未认领的 210 个框
根本不在 `_assigned` 里**，所以必须回到 `stage1`）。

顺带量：邻域有框但我们的 `desig_slots` 是空的那些（= 纯粹的**漏归属**）。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_desigdetect.py
    python work\\tools\\diag\\_desigdetect.py --pad 8 --split 200_train_cases
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES,             # noqa: E402
                                      assign_components, canon_type)
import stage2_eval as SE                                                # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"


def _inter(a, b):
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--s1", default=str(RUN / "stage1"))
    ap.add_argument("--asg", default=str(RUN / "stage2" / "_assigned"))
    ap.add_argument("--split", default="")
    ap.add_argument("--pad", type=float, default=6.0)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]

    st_pin = collections.Counter()
    hist = collections.Counter()          # 邻域文字框个数分桶
    hist_pin = collections.Counter()
    hist_iou = collections.Counter()      # 有框时：最好的相交比例分桶
    sam = collections.defaultdict(list)
    worst = collections.Counter()         # 每 case 的 A 类引脚数

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None or (a.split and case.split != a.split):
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        try:
            s1 = json.loads((Path(a.s1) / ("%s.stage1.json" % f.stem))
                            .read_text(encoding="utf-8"))
            asg = json.loads((Path(a.asg) / ("%s.json" % f.stem))
                             .read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            continue
        slots = asg.get("desig_slots") or {}
        tboxes = [t.get("bbox_img") for t in (s1.get("texts") or [])
                  if t.get("bbox_img")]
        # GT 框 -> 重叠最多的 stage1 元件 id（JSON 空间，见 `_desigcand2.py` 的坑）
        gcomps = [c for c in (s1.get("components") or [])]

        try:
            _f1, k2g = assign_components(gt, pr, cfg, {})
        except Exception:                                    # noqa: BLE001
            k2g = {}
        matched_g = set(k2g.values())

        for gk, gv in (gt.get("components") or {}).items():
            if canon_type((gv or {}).get("type")) not in IC_TYPES:
                continue
            if gk in matched_g:
                continue
            gp = len((gt.get("pins") or {}).get(gk) or {})
            st_pin["unmatched"] += 1
            st_pin["unmatched_pin"] += gp
            gb = list(gv.get("bbox") or [0, 0, 0, 0])
            pad = a.pad
            gbx = [gb[0] - pad, gb[1] - pad, gb[2] + pad, gb[3] + pad]
            garea = max(1.0, (gb[2] - gb[0]) * (gb[3] - gb[1]))
            near, bestiou = 0, 0.0
            for tb in tboxes:
                ov = _inter(gbx, tb)
                if ov <= 0:
                    continue
                near += 1
                bestiou = max(bestiou, ov / garea)
            # 取 X = 重叠最多的 stage1 元件，看它的 desig_slots 空不空
            bo, bi = 0.0, None
            for c in gcomps:
                ov = _inter(gb, c.get("bbox_json") or c.get("bbox_img")
                            or [0, 0, 0, 0])
                if ov > bo:
                    bo, bi = ov, c.get("id")
            empty_slot = not (slots.get(bi) or []) if bi else True

            bucket = "0" if near == 0 else ("1" if near == 1 else
                                            ("2-3" if near <= 3 else "4+"))
            hist[bucket] += 1
            hist_pin[bucket] += gp
            if near:
                ib = ("~0" if bestiou < 0.05 else "0.05-0.2" if bestiou < 0.2
                      else "0.2-0.5" if bestiou < 0.5 else "0.5+")
                hist_iou[ib] += 1
                if empty_slot:
                    st_pin["邻域有框但候选表空"] += 1
                    st_pin["邻域有框但候选表空_pin"] += gp
                if len(sam["B"]) < a.top:
                    sam["B"].append((case.case_id, gk, gp, bi, near,
                                     round(bestiou, 3), empty_slot))
            else:
                st_pin["A_邻域一个框都没有"] += 1
                st_pin["A_pin"] += gp
                worst[case.case_id] += gp
                if len(sam["A"]) < a.top:
                    sam["A"].append((case.case_id, gk, gp, bi,
                                     [round(x, 1) for x in gb]))

    def pct(n, d):
        return "%6.1f%%" % (100.0 * n / max(1, d))

    N, P = st_pin["unmatched"], st_pin["unmatched_pin"]
    print("== 没配上的 GT IC（%d 颗 / %d 引脚）：邻域（pad=%g）有没有 stage1 文字框 =="
          % (N, P, a.pad))
    print("  A 一个都没有  %5d %s  引脚 %5d %s   <- 检测/分割问题"
          % (st_pin["A_邻域一个框都没有"],
             pct(st_pin["A_邻域一个框都没有"], N),
             st_pin["A_pin"], pct(st_pin["A_pin"], P)))
    print("  B 有至少一个  %5d %s  引脚 %5d %s   <- 归属/识别问题"
          % (N - st_pin["A_邻域一个框都没有"],
             pct(N - st_pin["A_邻域一个框都没有"], N),
             P - st_pin["A_pin"], pct(P - st_pin["A_pin"], P)))
    print("      ·其中我们的候选表是**空的**（纯漏归属）%5d 引脚 %5d %s"
          % (st_pin["邻域有框但候选表空"], st_pin["邻域有框但候选表空_pin"],
             pct(st_pin["邻域有框但候选表空_pin"], P)))
    print("\n  邻域文字框个数分布（颗数 / 引脚）：")
    for k in ("0", "1", "2-3", "4+"):
        print("    %-4s %5d / %5d" % (k, hist[k], hist_pin[k]))
    print("\n  有框时「最好相交比 / GT 框面积」分布：")
    for k in ("~0", "0.05-0.2", "0.2-0.5", "0.5+"):
        if hist_iou[k]:
            print("    %-9s %5d" % (k, hist_iou[k]))
    if worst:
        print("\n  A 类最惨 case（引脚数）：", worst.most_common(12))
    for k in ("A", "B"):
        if sam[k]:
            print("\n%s 样例（case, GT键, 引脚, X, 邻域框数, 最好iou, 候选表空）：" % k)
            for s in sam[k]:
                print("   %s" % (s,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
