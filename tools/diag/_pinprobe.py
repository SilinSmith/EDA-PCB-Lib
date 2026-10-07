# -*- coding: utf-8 -*-
"""PinF1 为什么只有 0.039？（权重 25，是四个指标里最大的窟窿）

官方口径（`schematic_parser/metrics.py`）：
* `pin_f1`（`:430` 附近）把 `pins` 摊平成 `[(元件, 引脚号)] -> info`，键**必须完全相等**
  才算「同一个引脚」；然后要求 `pinname` 相等、`point` 距离 <= `pin_tol`（默认查 EvalConfig）。
* 非 box 类型的引脚不参与评分（Q&A 评测 Q2）。所以分母 = GT 里 box 元件的引脚数。

本脚本回答：
1. GT box 引脚的**引脚号长什么样**（`1`/`A1`/`GND1`…），我们预测的长什么样 —— 键根本对不上
   还是对上了但名字/坐标错。
2. 对我们**已成对**的 box 元件，GT 有几个引脚、我们写了几个引脚。

用法：python work\\tools\\diag\\_pinprobe.py --case 0001 [--pred work\\stage2_v14]
     python work\\tools\\diag\\_pinprobe.py --all --limit 20
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

EDA = r"Y:\workspace\EDA"
sys.path.insert(0, os.path.join(EDA, "eda-pcb"))
sys.path.insert(0, os.path.join(EDA, "work", "tools", "eval"))

from schematic_parser import metrics as M                      # noqa: E402
from schematic_parser.data import find_case, load_json         # noqa: E402
import stage2_eval                                             # noqa: E402

IC = M.IC_TYPES if hasattr(M, "IC_TYPES") else None
if IC is None:
    from schematic_parser.data import IC_TYPES as IC           # noqa: E402


def _flat_pins(section):
    """照抄 `metrics.pin_f1` 的摊平逻辑：`doc['pins'][元件][pin_键] -> info`。"""
    out = {}
    for comp, group in (section or {}).items():
        if isinstance(group, dict):
            for pk, info in group.items():
                out[(comp, M.pin_key_to_number(pk))] = info
        elif isinstance(group, list):
            for it in group:
                if isinstance(it, dict) and "pin" in it:
                    out[(comp, M.pin_key_to_number(it["pin"]))] = it
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--pred", default=os.path.join(EDA, "work", "stage2_v14"))
    a = ap.parse_args()

    cases = [a.case] if a.case else []
    if a.all:
        cases = []
        for f in sorted(os.listdir(a.pred)):
            if f.endswith(".json"):
                cases.append(os.path.splitext(f)[0])
        cases = cases[: a.limit]

    gk = Counter()      # GT box 引脚号形态
    pk = Counter()      # 我们 box 引脚号形态
    fb = [0, 0]         # 我们 box 引脚里 [回退的, 真读到号的]
    per = []            # (case, key, gt_pin, pred_pin, 键相同)

    def shape(s):
        s = str(s)
        if re.fullmatch(r"\d+", s):
            return "纯数字"
        if re.fullmatch(r"[A-Za-z]+\d+", s):
            return "字母+数字"
        if re.fullmatch(r"\d+_[A-Za-z0-9]+", s):
            return "N_xxx(我们合成的)"
        return "其它"

    for case in cases:
        c = find_case(case)
        if c is None or not c.json_path:
            continue
        gt = load_json(c.json_path)
        pp = os.path.join(a.pred, case + ".json")
        if not os.path.exists(pp):
            continue
        pred = load_json(pp)
        gf, pf = _flat_pins(gt.get("pins")), _flat_pins(pred.get("pins"))

        # 哪些元件是 box
        gt_box = set()
        for comp, info in (gt.get("components") or {}).items():
            if isinstance(info, dict) and M.canon_type(info.get("type")) in IC:
                gt_box.add(comp)
        pred_box = set(pred.get("components") or {})

        for (comp, num) in gf:
            if comp in gt_box:
                gk[shape(num)] += 1
        for (comp, num) in pf:
            if comp in pred_box:
                pk[shape(num)] += 1
                # 回退（没读到号）的判据：pinname == 引脚号 且是纯数字
                info = pf[(comp, num)] or {}
                if str(info.get("pinname")) == str(num) and re.fullmatch(r"\d+", str(num)):
                    fb[0] += 1
                else:
                    fb[1] += 1

        # 成对的 box 元件：官方是按「键（位号）相等」配的，这里直接按键配
        pcomp = pred.get("components") or {}
        pbykey = {}
        for cname, cinfo in pcomp.items():
            if isinstance(cinfo, dict):
                pbykey.setdefault(cinfo.get("key") or cname, cinfo)
        for gkey in gt_box:
            gi_ = (gt.get("components") or {}).get(gkey) or {}
            if gi_.get("key") and gi_["key"] != gkey:
                gkey2 = gi_["key"]
            else:
                gkey2 = gkey
            pc = pbykey.get(gkey2)
            if pc is None:
                continue
            pkey = pc.get("key") or gkey2
            gpin = [n for (cc, n) in gf if cc == gkey2]
            ppin = [n for (cc, n) in pf if cc == pkey]
            per.append((case, gkey2, len(gpin), len(ppin),
                        sum(1 for n in gpin if (pkey, n) in pf)))

    print("GT box 引脚号形态:", dict(gk.most_common()))
    print("我们 box 引脚号形态:", dict(pk.most_common()))
    print(f"我们 box 引脚：回退(没读到号、pinname==号) {fb[0]}，真读到号 {fb[1]}"
          f"  = 读到率 {fb[1]/max(1,fb[0]+fb[1]):.3f}")
    print()
    if per:
        tot_g = sum(x[2] for x in per)
        tot_p = sum(x[3] for x in per)
        tot_hit = sum(x[4] for x in per)
        print(f"成对的 box 元件 {len(per)} 个：GT 引脚 {tot_g}，我们写 {tot_p}，键完全相同 {tot_hit}"
              f"  = {tot_hit/max(1,tot_g):.3f}")
        print("样例：")
        for row in per[:30]:
            print(f"   {row[0]:>6s} {row[1]:>6s}  GT {row[2]:>3d} pin  我们 {row[3]:>3d} pin  键同 {row[4]:>3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
