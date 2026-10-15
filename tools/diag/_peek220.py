# -*- coding: utf-8 -*-
"""看一眼 work/_fast_base/*.json 的顶层结构，判断是「行列表」还是别的形状。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"Y:\workspace\EDA\work\_fast_base")

for name in sys.argv[1:]:
    p = BASE / (name + ".json")
    if not p.exists():
        print("== %s 不存在" % name)
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    print("== %s : %s" % (name, type(d).__name__))
    if isinstance(d, dict):
        for k, v in list(d.items())[:8]:
            if isinstance(v, list):
                print("   %-28s list[%d]" % (k, len(v)))
                if v and isinstance(v[0], dict):
                    print("      [0] keys: %s" % list(v[0].keys())[:14])
                    print("      [0] %s" % json.dumps(v[0], ensure_ascii=False)[:300])
            elif isinstance(v, dict):
                print("   %-28s dict%s" % (k, list(v.keys())[:10]))
            else:
                print("   %-28s %s" % (k, v))
    elif isinstance(d, list):
        print("   len=%d" % len(d))
        if d and isinstance(d[0], dict):
            print("   [0] keys: %s" % list(d[0].keys())[:16])
            print("   [0] %s" % json.dumps(d[0], ensure_ascii=False)[:400])
