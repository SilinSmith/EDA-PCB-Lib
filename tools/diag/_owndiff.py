# -*- coding: utf-8 -*-
"""对比同一张图、两个 `STAGE1_OWN` 设定下的 stage1 产物，看**归属闸到底并掉了谁**。

    python work\\tools\\diag\\_owndiff.py <off.json> <on.json>

判据：组件框（`bbox_img`）的角点距离 <=1px 视为同一个。输出被并掉的那些，
带上类型、框、分数、来源，以及它离最近同类 GT 有多远（用来判断是真重复还是误伤）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
sys.path.insert(0, str(ROOT / "eda-pcb"))


def load(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    out = {}
    for c in d.get("components") or []:
        b = c.get("bbox_img") or []
        if len(b) != 4:
            continue
        out[(round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1))] = c
    return d, out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    _, off = load(sys.argv[1])
    don, on = load(sys.argv[2])
    case = don.get("case") or "?"
    print("case %s：off 元件 %d，on 元件 %d" % (case, len(off), len(on)))
    gone = [c for k, c in off.items() if k not in on]
    new = [c for k, c in on.items() if k not in off]
    print("\n被并掉的 %d 个：" % len(gone))
    for c in gone:
        print("  %-6s %-14s score=%-6s src=%-18s box=%s"
              % (c["id"], c.get("type"), c.get("score"), c.get("source"), c.get("bbox_img")))
    print("凭空多出来的 %d 个：" % len(new))
    for c in new:
        print("  %-6s %-14s score=%-6s src=%-18s box=%s"
              % (c["id"], c.get("type"), c.get("score"), c.get("source"), c.get("bbox_img")))

    own = don.get("ownership") or {}
    print("\non 的 ownership: rule=%s demote=%s n_dropped=%s"
          % (own.get("rule"), own.get("demote"), own.get("n_dropped")))
    for e in own.get("ledger") or []:
        print("  并 %s(%.3f) <- %s(%.3f)  kind=%s dev=%s frac=%s rule=%s"
              % (e.get("kept"), e.get("kept_score") or 0, e.get("dropped"),
                 e.get("dropped_score") or 0, e.get("kind"), e.get("dev"),
                 e.get("frac"), e.get("rule")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
