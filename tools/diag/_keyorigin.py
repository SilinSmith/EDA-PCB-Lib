# -*- coding: utf-8 -*-
"""`U580` 是哪来的：把 r36 的最终文档键 与 `_assigned` 的 cid/位号 摆在一起。

用法（workdir = Y:\\workspace\\EDA）：python work\\tools\\diag\\_keyorigin.py [case]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.stdout.reconfigure(encoding="utf-8")
RUN = ROOT / "work" / "_submit_out220r36" / "runs" / "stage2"


def main(argv):
    case = (argv[0] if argv else "0001").zfill(4)
    out = json.loads((RUN / ("%s.json" % case)).read_text(encoding="utf-8"))
    asg = json.loads((RUN / "_assigned" / ("%s.json" % case)).read_text(encoding="utf-8"))
    keys = list(out["components"])
    print("最终文档 %d 个键：%s" % (len(keys), keys))
    print("\n_assigned: comps=%d desig_slots=%d n_free=%s n_owned=%s n_texts=%s"
          % (len(asg["comps"]), len(asg["desig_slots"]),
             asg.get("n_free"), asg.get("n_owned"), asg.get("n_texts")))
    ds = [(cid, (c.get("designator") or {}).get("text"))
          for cid, c in asg["comps"].items()]
    have = [x for x in ds if x[1]]
    print("有 designator 的 cid：%d 个 -> %s" % (len(have), have[:40]))
    print("无 designator 的 cid：%s" % [x[0] for x in ds if not x[1]])
    want = "U580"
    print("\n'%s' 在最终键里？%s；在 _assigned 的位号文本里？%s"
          % (want, want in keys,
             any(want == str(t) for _c, t in ds)))
    # 结构化槽位
    ss = asg.get("struct_slots") or {}
    print("\nstruct_slots 有 %d 项，样例 %s"
          % (len(ss), json.dumps(list(ss.items())[:1], ensure_ascii=False)[:400]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
