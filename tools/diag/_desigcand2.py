# -*- coding: utf-8 -*-
"""「正确位号到底在不在候选表里」—— 决定「修位号选择」这条路的**天花板**。

背景（`_icmisswhy.py` 已量）：220 张里没配上的 GT IC 有 579 颗 / 4587 引脚是
**R2_GT真位号读不出**，占全部引脚 fn 的 88%。但 `_u1probe.py` 在 case 0001 上
发现：正确位号 `U1` **一直在** `desig_slots` 里，而且 `conf` 0.983 远高于胜出的
`US8_D+`（conf 0.584）—— 只是**几何 `score`** 让不可能的形状赢了。

本脚本把这个单点发现推广到 220 张，量三件事：

  ① 对**每一颗** GT IC，找几何上重叠的我方元件 `cid`，它的 `desig_slots` 里
     有没有 `canon == GT键` 的候选？（= 修选择的**天花板**）
  ② 有的话，那条候选的 `score`/`conf` 相对胜出者是什么关系？
  ③ 没配上（R2）的那些，有多少是「候选表里有、我们没选它」。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_desigcand2.py
    python work\\tools\\diag\\_desigcand2.py --split 200_train_cases --top 15
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
                                      assign_components, canon_type, is_unkeyed)
import stage2_eval as SE                                                # noqa: E402
from fusion import claims as CL                                         # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"


def _overlap(a, b):
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x * y


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--s1", default=str(RUN / "stage1"))
    ap.add_argument("--asg", default=str(RUN / "stage2" / "_assigned"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]

    st = collections.Counter()       # 总计数
    st_pin = collections.Counter()
    rel = collections.Counter()      # 正确候选 vs 胜出者的置信度关系
    sam = collections.defaultdict(list)
    chosen_bad = collections.Counter()

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
        km = CL.vlm_keymap(s1, asg)                  # 装配键 -> cid
        slots = asg.get("desig_slots") or {}
        try:
            _f1, k2g = assign_components(gt, pr, cfg, {})
        except Exception:                                    # noqa: BLE001
            k2g = {}
        matched_g = set(k2g.values())
        pcomps = pr.get("components") or {}
        free_p = [(k, v) for k, v in pcomps.items() if k not in k2g]

        for gk, gv in (gt.get("components") or {}).items():
            if canon_type((gv or {}).get("type")) not in IC_TYPES:
                continue
            gp = len((gt.get("pins") or {}).get(gk) or {})
            st["gt_ic"] += 1
            st_pin["gt_ic"] += gp
            if gk in matched_g:
                st["matched"] += 1
                st_pin["matched"] += gp
                continue
            st["unmatched"] += 1
            st_pin["unmatched"] += gp

            gb = gv.get("bbox") or [0, 0, 0, 0]
            best, bestk = 0.0, None
            for pk, pv in free_p:
                ov = _overlap(gb, pv.get("bbox") or [0, 0, 0, 0])
                if ov > best:
                    best, bestk = ov, pk
            # ★★ 只差大小写：`metrics.assign_components:311` 的键匹配是
            # `set(gc) & set(pc)` —— **精确字符串**。GT 键 `u2` 与我方 `U2`
            # 因此配不上，而 `u2` 又不是 `∅NNN`（`is_unkeyed` 为假）⇒ 连坐标
            # 匹配的机会都没有，这颗 IC 的**全部引脚**直接判 fn。
            # 这一项如果大，就是一条极便宜的修法（键统一大小写）。
            ci = [k for k, _v in free_p
                  if str(k).lower() == str(gk).lower() and str(k) != str(gk)]
            if ci:
                st["只差大小写"] += 1
                st_pin["只差大小写"] += gp
                if len(sam["case_hit"]) < a.top:
                    sam["case_hit"].append((case.case_id, gk, gp, ci[:3]))
            if bestk is None:                        # R0：连重叠都没有
                st["R0"] += 1
                continue
            cid = km.get(str(bestk))
            # ★★ 坐标空间是踩过的坑：`stage1` 里每个元件有**两个**框 ——
            # `bbox_img`（图内/未翻转）与 `bbox_json`（装配后的 JSON 空间，y 已翻转）。
            # `assemble.build_doc` 发出的 `components[key]["bbox"]` **逐位等于
            # `bbox_json`**（实测 case 0001 的 c0000：`bbox_img=[1062.71,479.99,
            # 1146.23,563.63]` vs `bbox_json=[1062.71,526.37,1146.23,610.01]`，
            # 而文档里正是后者）。GT 的 `bbox` 也在 JSON 空间 ⇒ 拿 GT 框去和
            # `bbox_img` 求交会**大面积假零重叠**，cid 也就选错/选不到。
            bi, bo = None, 0.0
            for c in (s1.get("components") or []):
                ov = _overlap(gb, c.get("bbox_json") or c.get("bbox_img")
                              or [0, 0, 0, 0])
                if ov > bo:
                    bo, bi = ov, c.get("id")
            if bi is None:
                st["cid未知"] += 1
                continue
            if km.get(str(bestk)) != bi:
                st["bbox定cid与键不一致"] += 1
            cid = bi
            cands = slots.get(cid) or []
            # 我们**发出**的键
            d = ((asg.get("comps") or {}).get(cid) or {}).get("designator") or {}
            our = d.get("canon") or d.get("text")
            hit = [c for c in cands if str(c.get("canon")) == str(gk)]
            anyc = bool(cands)
            st["有候选表"] += 1 if anyc else 0
            if hit:
                st["候选表里有正确位号"] += 1
                st_pin["候选表里有正确位号"] += gp
                if is_unkeyed(gk):
                    rel["GT是占位符"] += 1
                else:
                    rel["GT是真位号"] += 1
                    h = max(hit, key=lambda c: float(c.get("score") or 0))
                    top = max(cands, key=lambda c: float(c.get("score") or 0))
                    hs, hc = float(h.get("score") or 0), float(h.get("conf") or 0)
                    ts, tc = float(top.get("score") or 0), float(top.get("conf") or 0)
                    rel["几何分不敌" if hs < ts else "几何分其实赢"] += 1
                    rel["置信度更高" if hc > tc else "置信度更低"] += 1
                    chosen_bad[str(top.get("canon"))] += 1
                    if len(sam["hit"]) < a.top:
                        sam["hit"].append(
                            (case.case_id, gk, gp, bestk, our,
                             [str(c.get("canon")) for c in cands][:5],
                             (round(hs, 3), round(hc, 3)), (round(ts, 3), round(tc, 3))))
            else:
                st["候选表里没有"] += 1
                st_pin["候选表里没有"] += gp
                # ★★ 关键细分：**根本没有候选**（该元件的邻域没读出任何位号 ——
                # 检测/关联问题）还是**有候选但全错**（读错了 —— 识别问题）。
                # 这两条要走的修法完全不同，不能混在一个「候选表里没有」里。
                if not anyc:
                    st["·候选表是空的"] += 1
                    st_pin["·候选表是空的"] += gp
                else:
                    st["·有候选但全错"] += 1
                    st_pin["·有候选但全错"] += gp
                if len(sam["miss"]) < a.top:
                    sam["miss"].append(
                        (case.case_id, gk, gp, bestk, our,
                         [(str(c.get("canon")), round(float(c.get("score") or 0), 2),
                           round(float(c.get("conf") or 0), 2)) for c in cands][:5]))

    def pct(n, d):
        return "%6.1f%%" % (100.0 * n / max(1, d))

    print("== GT IC 总体 ==")
    print("  元件 %d（配上 %d / 没配上 %d）  引脚 %d（配上 %d / 没配上 %d）"
          % (st["gt_ic"], st["matched"], st["unmatched"],
             st_pin["gt_ic"], st_pin["matched"], st_pin["unmatched"]))
    print("\n== 没配上的那些，正确位号在不在候选表里 ==")
    print("  找到重叠元件并查到 cid      %5d / %d" % (st["有候选表"], st["unmatched"]))
    print("  候选表里**有**正确位号      %5d %s   引脚 %5d %s"
          % (st["候选表里有正确位号"],
             pct(st["候选表里有正确位号"], st["unmatched"]),
             st_pin["候选表里有正确位号"],
             pct(st_pin["候选表里有正确位号"], st_pin["unmatched"])))
    print("  候选表里没有                %5d %s   引脚 %5d %s"
          % (st["候选表里没有"], pct(st["候选表里没有"], st["unmatched"]),
             st_pin["候选表里没有"], pct(st_pin["候选表里没有"], st_pin["unmatched"])))
    print("      ·其中「候选表是空的」    %5d   引脚 %5d  %s"
          % (st["·候选表是空的"], st_pin["·候选表是空的"],
             pct(st_pin["·候选表是空的"], st_pin["unmatched"])))
    print("      ·其中「有候选但全错」    %5d   引脚 %5d  %s"
          % (st["·有候选但全错"], st_pin["·有候选但全错"],
             pct(st_pin["·有候选但全错"], st_pin["unmatched"])))
    print("  连重叠元件都没有(R0)        %5d" % st["R0"])
    print("  ★ 只差大小写的键            %5d   引脚 %5d  %s"
          % (st["只差大小写"], st_pin["只差大小写"],
             pct(st_pin["只差大小写"], st_pin["unmatched"])))
    print("  bbox 反查补上的 cid         %5d" % st["bbox反查"])
    print("  cid 查不到                  %5d" % st["cid未知"])
    print("\n== 「候选表里有正确位号」细分 ==")
    for k in ("GT是真位号", "GT是占位符", "几何分不敌", "几何分其实赢",
              "置信度更高", "置信度更低"):
        if rel[k]:
            print("  %-14s %5d" % (k, rel[k]))
    if chosen_bad:
        print("  胜出者（错）的常见值：", chosen_bad.most_common(12))
    for k in ("hit", "miss"):
        if sam[k]:
            print("\n%s 样例（case, GT键, GT引脚, 我方键, 我方发出, 候选, 正确(score,conf), 胜出(score,conf)）：" % k)
            for s in sam[k]:
                print("   %s" % (s,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
