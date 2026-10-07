# -*- coding: utf-8 -*-
"""探针：case 0001 里 GT 的 U1 对应我们哪颗元件、那颗元件的位号候选表长什么样。

用法（workdir = Y:\\workspace\\EDA）：python work\\tools\\diag\\_u1probe.py [case]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))
sys.path.insert(0, str(ROOT / "work" / "tools" / "eval"))
sys.stdout.reconfigure(encoding="utf-8")

from schematic_parser.data import DEFAULT_DATA_ROOT, load_json  # noqa: E402
from schematic_parser.metrics import EvalConfig, assign_components  # noqa: E402
import stage2_eval as SE  # noqa: E402

RUN = ROOT / "work" / "_submit_out220r36" / "runs" / "stage2"


def ov(a, b):
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def main(argv):
    want_case = (argv[0] if argv else "0001").zfill(4)
    exact, digits = SE.build_index(DEFAULT_DATA_ROOT)
    case, _ = SE.resolve(want_case, exact, digits)
    gt = load_json(case.json_path)
    pr = json.loads((RUN / ("%s.json" % want_case)).read_text(encoding="utf-8"))
    asg = json.loads((RUN / "_assigned" / ("%s.json" % want_case))
                     .read_text(encoding="utf-8"))
    _f1, km = assign_components(gt, pr, EvalConfig(), {})
    print("keymap（我方键 -> GT 键）= %s" % km)
    gb = gt["components"]["U1"]["bbox"]
    print("GT U1 bbox = %s，引脚 %d 个" % (gb, len(gt["pins"].get("U1") or {})))
    scored = sorted(((ov(gb, v.get("bbox") or [0, 0, 0, 0]), k)
                     for k, v in pr["components"].items()), reverse=True)
    for o, k in scored[:3]:
        print("\n我方 **键** %s 重叠 %.0f  type=%s bbox=%s"
              % (k, o, pr["components"][k].get("type"),
                 [round(x, 1) for x in (pr["components"][k].get("bbox") or [])]))
    # ★ `_assigned` 的键是 **cid**（c0000…），不是最终键 —— 先按框重叠找回 cid
    print("\n== 按元件的 bbox 与 GT U1 的重叠找 cid ==")
    cs = sorted(((ov(gb, v.get("bbox") or [0, 0, 0, 0]), cid)
                 for cid, v in asg["comps"].items()), reverse=True)
    for o, cid in cs[:3]:
        c = asg["comps"][cid]
        print("\ncid=%s 重叠 %.0f  bbox=%s" % (cid, o,
              [round(x, 1) for x in (c.get("bbox") or [])]))
        print("   designator = %s"
              % json.dumps(c.get("designator"), ensure_ascii=False)[:300])
        print("   desig_slots = %s"
              % json.dumps(asg["desig_slots"].get(cid), ensure_ascii=False)[:700])
    print("\n== 哪些 cid 的 designator.text 里含 '580' ==")
    for cid, c in asg["comps"].items():
        d = c.get("designator") or {}
        if "580" in str(d.get("text") or ""):
            print("   %s -> %s" % (cid, json.dumps(d, ensure_ascii=False)[:300]))
    print("\n== designator 为空的 cid 数 = %d / %d =="
          % (sum(1 for c in asg["comps"].values()
                 if not (c.get("designator") or {}).get("text")),
             len(asg["comps"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
