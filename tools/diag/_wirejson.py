# -*- coding: utf-8 -*-
"""_wirejson.py —— 打印预测/GT 的 nets 段原样，确认 edges 里点的形状。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]

for cid, p in (("0048", ROOT / "work/_f220_A/0048.json"),
               ("0036", ROOT / "work/_f220_A/0036.json"),
               ("0092", ROOT / "work/_f220_A/0092.json")):
    pr = json.loads(Path(p).read_text(encoding="utf-8"))
    nets = pr.get("nets") or {}
    print("=== %s  nets=%d" % (cid, len(nets)))
    for k, v in list(nets.items())[:3]:
        print("  net %r keys=%s" % (k, list(v.keys())))
        print("    raw=%s" % json.dumps(v, ensure_ascii=False)[:700])
    # 统计各形状
    shapes = {}
    for k, v in nets.items():
        for ek, e in (v.get("edges") or {}).items():
            kk = type(e).__name__
            if isinstance(e, list) and e:
                kk += "[%s]" % type(e[0]).__name__
                if isinstance(e[0], dict):
                    kk += "keys=%s" % ",".join(sorted(e[0].keys()))
            shapes[kk] = shapes.get(kk, 0) + 1
    print("  edges 形状直方图: %s" % shapes)
