# -*- coding: utf-8 -*-
"""逐脚对账：我们的某个元件 vs GT —— 号对不对、名对不对。只读。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.stdout.reconfigure(encoding="utf-8")


def gt_doc(case: str):
    hits = sorted(glob.glob(str(ROOT / "eda-pcb" / "yolo_part"
                                / "Dataset(waitforpreprocess)" / "*" / ("*%s" % case)
                                / "*_target_named.json")))
    if not hits:
        raise SystemExit("找不到 case %s 的 GT" % case)
    d = json.loads(Path(hits[0]).read_text(encoding="utf-8"))
    return d, Path(hits[0]).name


def main(argv=None):
    case = argv[0] if argv else "0143"
    pred_path = argv[1] if len(argv) > 1 else str(
        ROOT / "work" / "_pin1_iter" / "runs" / "fusion" / ("%s.json" % case))
    key = argv[2] if len(argv) > 2 else "U1"
    g, gname = gt_doc(case)
    d = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    types = {}
    for k, v in (g.get("components") or {}).items():
        types[k] = (v or {}).get("type")
    print("GT %s  元件类型 %s" % (gname, {k: types[k] for k in list(types)[:30]}))
    gp = (g.get("pins") or {}).get(key) or {}
    pp = (d.get("pins") or {}).get(key) or {}
    gn = {str(k).replace("pin_", ""): (v or {}).get("pinname") for k, v in gp.items()}
    pn = {str(k).replace("pin_", ""): (v or {}).get("pinname") for k, v in pp.items()}
    hit = sum(1 for k, v in pn.items() if gn.get(k) == v)
    both = sum(1 for k, v in pn.items() if k in gn)
    print("GT %d 脚 · 我们 %d 脚 · 键同时命中 %d · 键+名都对 %d"
          % (len(gn), len(pn), both, hit))
    # 只按名字比（不管号）—— 看名字本身读得对不对
    by_name = sum(1 for v in pn.values() if v in set(gn.values()))
    print("只看名字：我们 %d 个名字里有 %d 个在 GT 的名字集合里" % (len(pn), by_name))
    order = sorted(set(list(gn) + list(pn)),
                   key=lambda s: (0, int(s)) if str(s).isdigit() else (1, str(s)))
    for k in order:
        a, b = gn.get(k), pn.get(k)
        mark = "OK " if (a is not None and a == b) else ("~~ " if a is None else "XX ")
        print("%s GT %-8s %-20s | 我们 %-20s" % (mark, k, a if a is not None else "<无>",
                                                 b if b is not None else "<无>"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
