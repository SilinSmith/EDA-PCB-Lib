# -*- coding: utf-8 -*-
"""引脚 tp 到底丢在哪一关：**键 → 号 → 名 → 位置** 四关漏斗。

官方 `pin_f1`（`eda-pcb\\schematic_parser\\metrics.py:426-464`）的判据是四关全过才算 tp：

  ① 元件键要配上   —— `set(gc) & set(pc)` 精确字符串（`:311`），配上后再用 keymap 改写我方键
  ② 引脚号要配上   —— `out[(comp, pin_key_to_number(pk))]`（`:442`）
  ③ `pinname` 要**逐字相等** —— `if p.get('pinname') != g.get('pinname'): fp+=1; fn+=1`（`:455`）
  ④ 点要在 `cfg.pin_tol`（=5.0 px）内 —— `math.hypot(...) > cfg.pin_tol`（`:460`）

本脚本按这四关**逐关放行**，量出每一关牺牲了多少 tp，并把第③关的失配原样打出来
—— 如果失配只是「分隔符 / 大小写 / 多一个空格 / 一侧为空」的差异，那么**一次归一化**
就能换来大量 tp，而这属于融合层（我方）完全可控的范围，比重新训练识别器便宜得多。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_pinfunnel.py
    python work\\tools\\diag\\_pinfunnel.py --pred work\\_fusion_out --top 25
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json          # noqa: E402
from schematic_parser.metrics import (EvalConfig, IC_TYPES,             # noqa: E402
                                      assign_components, canon_type,
                                      pin_key_to_number, _point)
import stage2_eval as SE                                                # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs"
_NORM = re.compile(r"[^0-9A-Za-z]")


def norm_name(s):
    """把 pinname 归一成「只留字母数字并大写」——用来判断失配是不是纯格式差异。"""
    return _NORM.sub("", str(s if s is not None else "")).upper()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUN / "stage2"))
    ap.add_argument("--split", default="")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)

    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    cfg = EvalConfig()
    files = [f for f in sorted(Path(a.pred).glob("*.json"))
             if not f.name.startswith("_")]

    st = collections.Counter()
    name_pairs = collections.Counter()      # (GT名, 我方名) 的常见组合
    num_pairs = collections.Counter()       # (GT号, 我方号) 的常见组合
    dists = []
    samples = []
    norm_would = collections.Counter()

    for f in files:
        case, _how = SE.resolve(f.stem, exact, digits)
        if case is None or (a.split and case.split != a.split):
            continue
        gt = load_json(case.json_path)
        pr = json.loads(f.read_text(encoding="utf-8"))
        try:
            _f1, k2g = assign_components(gt, pr, cfg, {})
        except Exception:                                    # noqa: BLE001
            continue
        # 官方口径：只统计 GT 里 type 为 box 的元件
        box_comps = {k for k, v in (gt.get("components") or {}).items()
                     if canon_type((v or {}).get("type")) in IC_TYPES}
        # 官方在跑 pin_f1 之前会用 keymap 改写我方键
        pk_pins = {}
        for pk, grp in (pr.get("pins") or {}).items():
            pk_pins[k2g.get(pk, pk)] = grp

        for gk in box_comps:
            ggrp = (gt.get("pins") or {}).get(gk) or {}
            pgrp = pk_pins.get(gk) or {}
            if not isinstance(ggrp, dict):
                continue
            st["gt_box_comps"] += 1
            if not ggrp:
                st["gt_comp_无引脚"] += 1
                continue
            st["gt_comp_有引脚"] += 1
            if not isinstance(pgrp, dict) or not pgrp:
                st["我方该元件无引脚"] += 1
            gnum = {pin_key_to_number(k): v for k, v in ggrp.items()}
            pnum = {pin_key_to_number(k): v for k, v in (pgrp or {}).items()}
            st["gt_pins"] += len(gnum)
            st["our_pins"] += len(pnum)
            # ②关失败分两种，修法完全不同：
            #   (a) 我方**整组引脚都没发**（assemble 里 `BOX_PIN_DROP_*` 把它整组丢了）
            #       ⇒ 得先让 stage2 敢发这一组
            #   (b) 我方发了引脚，只是**号对不上** ⇒ 是识别/分配问题
            if not pnum:
                st["(a)整组未发的元件"] += 1
                st["(a)整组未发的元件_GT引脚"] += len(gnum)
                st["(a)整组未发_号配上"] += len(set(gnum) & set(pnum))
            else:
                st["(b)发了引脚的元件"] += 1
                st["(b)发了引脚的元件_GT引脚"] += len(gnum)
                st["(b)发了引脚_号配上"] += len(set(gnum) & set(pnum))
                st["(b)发了引脚_多出来的号"] += len(set(pnum) - set(gnum))
            for num in set(gnum) | set(pnum):
                g, p = gnum.get(num), pnum.get(num)
                if g is None:
                    st["fp_多出来的号"] += 1
                    num_pairs[("<无>", str(num))] += 1
                    continue
                if p is None:
                    st["②号没配上"] += 1
                    num_pairs[(str(num), "<无>")] += 1
                    continue
                st["②号配上"] += 1
                gn = (g or {}).get("pinname")
                pn = (p or {}).get("pinname")
                if gn != pn:
                    st["③名不等"] += 1
                    name_pairs[(str(gn)[:14], str(pn)[:14])] += 1
                    if norm_name(gn) == norm_name(pn):
                        norm_would["归一化后相等"] += 1
                    elif not norm_name(gn) and not norm_name(pn):
                        norm_would["两边都空"] += 1
                    if len(samples) < a.top:
                        samples.append((case.case_id, gk, num,
                                        repr(gn)[:26], repr(pn)[:26],
                                        "归一化可救" if norm_name(gn) == norm_name(pn) else ""))
                    continue
                st["③名相等"] += 1
                pa, pb = _point((g or {}).get("point")), _point((p or {}).get("point"))
                if pa is None or pb is None:
                    st["④没有点"] += 1
                    continue
                d = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                dists.append(d)
                if d > cfg.pin_tol:
                    st["④位置超差"] += 1
                else:
                    st["★tp"] += 1

    def pct(n, d):
        return "%6.1f%%" % (100.0 * n / max(1, d))

    print("== 元素级 ==")
    print("  GT 里 box 元件 %d（有引脚 %d / 无引脚 %d）"
          % (st["gt_box_comps"], st["gt_comp_有引脚"], st["gt_comp_无引脚"]))
    print("  其中有引脚的元件里，我方**一条引脚都没有**的： %d  %s"
          % (st["我方该元件无引脚"],
             pct(st["我方该元件无引脚"], st["gt_comp_有引脚"])))
    print("\n== ②关失败的两种形态 ==")
    print("  (a) 我方整组引脚都没发：元件 %d  承载 GT 引脚 %5d  %s  其中号配上 %d"
          % (st["(a)整组未发的元件"], st["(a)整组未发的元件_GT引脚"],
             pct(st["(a)整组未发的元件_GT引脚"], st["gt_pins"]),
             st["(a)整组未发_号配上"]))
    print("  (b) 我方发了引脚但号对不上：元件 %d  承载 GT 引脚 %5d  %s  其中号配上 %5d  %s  多出来的号 %d"
          % (st["(b)发了引脚的元件"], st["(b)发了引脚的元件_GT引脚"],
             pct(st["(b)发了引脚的元件_GT引脚"], st["gt_pins"]),
             st["(b)发了引脚_号配上"],
             pct(st["(b)发了引脚_号配上"], st["(b)发了引脚的元件_GT引脚"]),
             st["(b)发了引脚_多出来的号"]))
    print("\n== 引脚漏斗（分母 = GT 引脚 %d） ==" % st["gt_pins"])
    print("  ② 号没配上（我方没有这个号）      %5d  %s"
          % (st["②号没配上"], pct(st["②号没配上"], st["gt_pins"])))
    print("  ② 号配上                        %5d  %s"
          % (st["②号配上"], pct(st["②号配上"], st["gt_pins"])))
    print("     ③ 名不等                      %5d  %s"
          % (st["③名不等"], pct(st["③名不等"], st["gt_pins"])))
    print("     ③ 名相等                      %5d  %s"
          % (st["③名相等"], pct(st["③名相等"], st["gt_pins"])))
    print("        ④ 没有点                   %5d" % st["④没有点"])
    print("        ④ 位置超差(>%.1f)          %5d" % (cfg.pin_tol, st["④位置超差"]))
    print("        ★ tp                       %5d  %s"
          % (st["★tp"], pct(st["★tp"], st["gt_pins"])))
    print("  我方多出来的号(fp)               %5d" % st["fp_多出来的号"])

    if dists:
        ds = sorted(dists)
        q = lambda p: ds[min(len(ds) - 1, int(len(ds) * p))]     # noqa: E731
        print("\n== 「③名相等」那批的点距离（n=%d） ==" % len(ds))
        print("  p10 %.1f  p25 %.1f  p50 %.1f  p75 %.1f  p90 %.1f  max %.1f"
              % (q(.10), q(.25), q(.50), q(.75), q(.90), ds[-1]))
        for tol in (2, 3, 5, 8, 12, 20):
            print("    tol<=%-3d  %5d  %s" % (tol, sum(1 for d in ds if d <= tol),
                                             pct(sum(1 for d in ds if d <= tol), len(ds))))

    print("\n== ③关失配能否用归一化救回 ==")
    for k in ("归一化后相等", "两边都空"):
        print("  %-12s %5d  （占③失配 %s）"
              % (k, norm_would[k], pct(norm_would[k], st["③名不等"])))
    if name_pairs:
        print("\n  (GT名, 我方名) 最常见的组合：")
        for (gn, pn), n in name_pairs.most_common(15):
            print("    %-16s -> %-16s %5d" % (gn, pn, n))
    if num_pairs:
        print("\n  (GT号, 我方号) 最常见的失配组合：")
        for (gn, pn), n in num_pairs.most_common(15):
            print("    %-10s -> %-10s %5d" % (gn, pn, n))
    if samples:
        print("\n  ③关失配逐条样例（case, 元件键, 号, GT名, 我方名, 备注）：")
        for s in samples:
            print("    %s" % (s,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
