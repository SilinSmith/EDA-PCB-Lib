# -*- coding: utf-8 -*-
"""看 `_assigned/<case>.json` 里**每个引脚槽位**到底带了什么（尤其：名字那段文字的框在不在）。

用途：判断「把引脚输出点从『框边接点』改成『引脚名文字的框心』」在数据上做不做得到。

    python work/tools/diag/_pinslot.py 1
    python work/tools/diag/_pinslot.py 1 --comp U1 --limit 8
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--run", default=str(ROOT / "work" / "_full220"))
    ap.add_argument("--comp")
    ap.add_argument("--limit", type=int, default=4)
    a = ap.parse_args()
    q = Path(a.run) / "runs" / "stage2" / "_assigned" / (a.case + ".json")
    d = json.load(open(q, encoding="utf-8"))
    print("文件：%s" % q)
    print("顶层键：%s" % list(d.keys()))
    comps = d.get("comps") or {}
    print("元件数：%d" % len(comps))
    shown = 0
    for cid, c in comps.items():
        if a.comp and cid != a.comp:
            continue
        pins = c.get("pins") or {}
        if not pins:
            continue
        print()
        print("=== %s type=%s desig=%r n_pins=%d ==="
              % (cid, c.get("type"), c.get("designator"), len(pins)))
        for i, p in list(pins.items())[:a.limit]:
            print("  slot %s -> %s" % (i, json.dumps(p, ensure_ascii=False)))
        shown += 1
        if shown >= 3:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
