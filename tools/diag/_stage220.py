# -*- coding: utf-8 -*-
"""把一次 `submission\\run.py` 全量跑的**关键阶段**逐个抽出来看（用户 m31144 的要求）。

逐阶段看什么：
  阶段一（YOLO + CV + 骨架）：每张的元件/文字框/引脚触点数/网络数，以及 `timing` 各段耗时。
  阶段二（文字识别 + 装配）：`runs/stage2/_assigned/<case>.json` 里
      * 位号有值 / 空；
      * 每颗元件的引脚槽位里 `number` 读出来的比例、`name` 读出来的比例
        —— **这是引脚分和网络分的共同源头**（`assemble.py:761 nm = r["nm"] or num`）。
  产物层：`final_results/<case>_Pred.json` 里 `pinname` 就是引脚号的比例。
  打分：复用 `stage2_eval` 的配对，打印四项 + 加权总分（`metrics.py:42 WEIGHTS`）。

用法：
    python work\\tools\\diag\\_stage220.py --run work\\_full220
    python work\\tools\\diag\\_stage220.py --run work\\_full220 --split 200_train_cases
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:\workspace\EDA")
EDA = ROOT / "eda-pcb"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from schematic_parser.data import DEFAULT_DATA_ROOT, discover, load_json   # noqa: E402
from schematic_parser.metrics import (EvalConfig, WEIGHTS, canon_type,      # noqa: E402
                                      IC_TYPES, evaluate_case, normalise_doc)

TIMING_KEYS = ["yolo_s", "cv_s", "wires_s", "rays_s", "text_s", "bigic_s", "pins_s", "total_s"]


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "work" / "_full220"))
    ap.add_argument("--split", default="")
    ap.add_argument("--worst", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    s1d, s2d, fdir = run / "runs" / "stage1", run / "runs" / "stage2" / "_assigned", run / "final_results"

    cases = discover(DEFAULT_DATA_ROOT)
    if a.split:
        cases = [c for c in cases if c.split == a.split]

    T = Counter()
    tim = {k: [] for k in TIMING_KEYS}
    per = []
    cfg = EvalConfig()
    for c in cases:
        p1 = s1d / (c.case_id + ".stage1.json")
        if not p1.exists():
            T["缺少 stage1"] += 1
            continue
        s1 = load_json(p1)
        st = s1.get("stats") or {}
        T["cases"] += 1
        T["components"] += int(st.get("n_components") or 0)
        T["texts"] += int(st.get("n_texts") or 0)
        T["pins"] += int(st.get("n_pins") or 0)
        T["nets"] += int(st.get("n_nets") or 0)
        T["pins_in_nets"] += int(st.get("n_pins_in_nets") or 0)
        for k in TIMING_KEYS:
            v = (s1.get("timing") or {}).get(k)
            if isinstance(v, (int, float)):
                tim[k].append(float(v))
        p2 = s2d / (c.case_id + ".json")
        if p2.exists():
            A = load_json(p2)
            for cid, cc in (A.get("comps") or {}).items():
                T["comps2"] += 1
                T["desig_ok"] += bool((cc.get("designator") or {}).get("text"))
                for _i, rec in ((cc.get("pins") or {}) or {}).items():
                    T["slots"] += 1
                    T["num"] += bool((rec.get("number") or {}).get("text"))
                    T["nm"] += bool((rec.get("name") or {}).get("text"))
        pr = None
        for nm in (c.case_id + "_Pred.json", c.case_id + ".json"):
            if (fdir / nm).exists():
                pr = load_json(fdir / nm)
                break
        if pr is None:
            continue
        for _comp, pins in (pr.get("pins") or {}).items():
            for pk, v in (pins or {}).items():
                T["out_pins"] += 1
                nm_ = str((v or {}).get("pinname") or "")
                T["out_name_is_num"] += (nm_ == pk.replace("pin_", "").split("_")[0])
                T["out_name_digit"] += nm_.isdigit()
        try:
            g = normalise_doc(load_json(Path(c.json_path)))
            r = evaluate_case(g, normalise_doc(pr), cfg)
            per.append((r.score(), c.case_id, c.split, r.component.f1, r.pin.f1,
                        r.net_hypergraph.f1, r.net_line.f1,
                        len(g.get("pins", {})), len(pr.get("pins", {}))))
        except Exception as e:                                  # noqa: BLE001
            T["打分失败"] += 1
            if T["打分失败"] < 3:
                print("  打分失败 %s: %s" % (c.case_id, e))

    n = max(1, T["cases"])
    print("== 跑目录 %s ==" % run)
    print("case %d（split=%s）" % (T["cases"], a.split or "全部"))
    print("\n-- 阶段一（视觉）逐张平均 --")
    for k in ("components", "texts", "pins", "nets", "pins_in_nets"):
        print("   %-14s 合计 %8d  平均 %8.1f" % (k, T[k], T[k] / n))
    print("\n-- 阶段一 耗时（秒）--")
    print("   %-10s %10s %10s %10s" % ("段", "合计", "平均", "中位"))
    for k in TIMING_KEYS:
        v = tim[k]
        print("   %-10s %10.1f %10.2f %10.2f" % (k, sum(v), sum(v) / max(1, len(v)), med(v)))
    print("\n-- 阶段二（装配）--")
    print("   元件 %d 个，位号有值 %d（%.1f%%）" % (
        T["comps2"], T["desig_ok"], 100.0 * T["desig_ok"] / max(1, T["comps2"])))
    print("   引脚槽位 %d 个：**读出号** %d（%.1f%%）、**读出名** %d（%.1f%%）" % (
        T["slots"], T["num"], 100.0 * T["num"] / max(1, T["slots"]),
        T["nm"], 100.0 * T["nm"] / max(1, T["slots"])))
    print("\n-- 产物层 --")
    print("   引脚 %d 个：`pinname` == 引脚号 %d（%.1f%%）、纯数字 %d（%.1f%%）" % (
        T["out_pins"], T["out_name_is_num"], 100.0 * T["out_name_is_num"] / max(1, T["out_pins"]),
        T["out_name_digit"], 100.0 * T["out_name_digit"] / max(1, T["out_pins"])))
    if per:
        m = [sum(p[i] for p in per) / len(per) for i in (0, 3, 4, 5, 6)]
        w = 100.0 * sum(WEIGHTS[k] * v for k, v in zip(
            ("component", "pin", "net_hypergraph", "net_line"), m[1:]))
        print("\n-- 打分（%d 张，均值为逐张均值）--" % len(per))
        print("   逐张均分 %.4f ｜ 元件 %.4f  引脚 %.4f  网络 %.4f  导线 %.4f" % (
            m[0], m[1], m[2], m[3], m[4]))
        print("   按**官方权重**（元件 0.30 / 引脚 0.25 / 网络 0.35 / 导线 0.10）加权 = **%.2f**" % w)
        if a.worst:
            per.sort()
            print("   最差 %d 张：" % a.worst)
            for s, cid, sp, cf, pf, nh, nl, gp, pp in per[:a.worst]:
                print("     %-10s %-16s %6.2f | 元件%.3f 引脚%.3f 网络%.3f 导线%.3f | GT引脚%d 我们%d"
                      % (cid, sp, s, cf, pf, nh, nl, gp, pp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
