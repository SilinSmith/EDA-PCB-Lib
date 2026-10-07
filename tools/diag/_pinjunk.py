# -*- coding: utf-8 -*-
"""量「IC 的预测引脚里有多少是垃圾」—— 用来判断 `pin_f1` 的 fp 有多少是白送的。

口径：只统计 GT 认定为 IC（`canon_type(type) in IC_TYPES`）的元件，因为官方 `pin_f1`
的 `box_comps` 就是这一批；非 IC 元件我们输出多少引脚都不影响分数。

三类垃圾的判据（都很保守，宁可漏判）：
  ① `名在型号里`：`pinname` 长度 ≥4 且（去掉 `/` 后）是元件 `Name` 的子串 ——
     这是把框下方印的型号 `MSP432E401YTPDT` 切碎当成引脚名读出来了。
  ② `长数字`：`pinname` 以数字开头且长度 ≥3（真引脚名几乎不会是 3 位以上纯数字开头）。
  ③ `GT 没这个键`：预测的 `pin_N` 键在原 GT 里不存在 ⇒ 必是 fp。

用法：`python work\\tools\\diag\\_pinjunk.py [--pred 目录] [--cases 1,2,3] [--show]`
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))

from stage2_eval import DEFAULT_DATA_ROOT, build_index, load_json, resolve  # noqa: E402
from schematic_parser.metrics import IC_TYPES, canon_type  # noqa: E402

FAST = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
        "0092", "0048", "0019", "0024", "0075", "0086", "0002", "0001", "0042", "0103"]


def pred_file(d: Path, cid: str) -> Path:
    for name in ("%s_Pred.json" % cid, "%s.json" % cid):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(cid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "work" / "_fast_out" / "final_results"))
    ap.add_argument("--cases", default="")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    cases = [x.strip() for x in a.cases.split(",") if x.strip()] or FAST
    exact, digits = build_index(DEFAULT_DATA_ROOT)

    tot = collections.Counter()
    rows = []
    for cid in cases:
        c, how = resolve(cid, exact, digits)
        gt = load_json(c.json_path)
        pr = json.loads(pred_file(pred_dir, cid).read_text(encoding="utf-8"))
        gt_ics = {k for k, v in (gt.get("components") or {}).items()
                  if canon_type(v.get("type")) in IC_TYPES}
        r = collections.Counter()
        r["gt_ics"] = len(gt_ics)
        for k in gt_ics:
            g = (gt.get("pins") or {}).get(k) or {}
            p = (pr.get("pins") or {}).get(k) or {}
            nm = str(((pr.get("components") or {}).get(k) or {}).get("Name") or "")
            r["gt_pins"] += len(g)
            r["pr_pins"] += len(p)
            for pk, pv in p.items():
                name = str((pv or {}).get("pinname") or "")
                junk = 0
                if len(name) >= 4 and nm and name.replace("/", "").upper() in nm.upper():
                    r["name_in_partnumber"] += 1
                    junk = 1
                if name[:1].isdigit() and len(name) >= 3:
                    r["numeric_long"] += 1
                    junk = 1
                if pk not in g:
                    r["key_absent"] += 1
                    junk = 1
                if junk:
                    r["junk"] += 1
                else:
                    r["ok"] += 1
        rows.append((cid, r))
        tot.update(r)

    print("%-8s %6s %7s %7s %7s %6s %6s %6s" %
          ("case", "ICs", "GT引脚", "PR引脚", "垃圾", "名在型号", "长数字", "键不存在"))
    for cid, r in rows:
        print("%-8s %6d %7d %7d %7d %6d %6d %6d" %
              (cid, r["gt_ics"], r["gt_pins"], r["pr_pins"], r["junk"],
               r["name_in_partnumber"], r["numeric_long"], r["key_absent"]))
    print()
    print("合计：GT 引脚 %d  预测引脚 %d  其中垃圾 %d (%.3f)"
          % (tot["gt_pins"], tot["pr_pins"], tot["junk"],
             tot["junk"] / max(1, tot["pr_pins"])))
    print("      名在型号里 %d ｜ 长数字 %d ｜ GT 没这个键 %d ｜ 看着正常 %d"
          % (tot["name_in_partnumber"], tot["numeric_long"], tot["key_absent"], tot["ok"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
