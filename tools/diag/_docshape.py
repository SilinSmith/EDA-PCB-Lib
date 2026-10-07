# -*- coding: utf-8 -*-
"""看最终文档里 `pins` / `nets` 的真实字段形状 —— 融合层要能改它们，得先知道改什么。

用法（workdir = Y:\\workspace\\EDA）：
    python work\\tools\\diag\\_docshape.py [case] [目录]
默认 case 0001、目录 work\\_fusion_out。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")


def main(argv):
    case = (argv[0] if argv else "0001").zfill(4)
    d = Path(argv[1]) if len(argv) > 1 else (ROOT / "work" / "_fusion_out")
    p = d / ("%s.json" % case)
    if not p.exists():
        print("没有 %s" % p)
        return 1
    doc = json.loads(p.read_text(encoding="utf-8"))
    print("顶层键：%s" % sorted(doc.keys()))
    comps = doc.get("components") or {}
    print("元件 %d 个；键样例 %s" % (len(comps), list(comps)[:12]))
    k0 = next(iter(comps), None)
    if k0:
        print("\ncomponents[%s] = %s" % (k0, json.dumps(comps[k0], ensure_ascii=False)[:600]))
    pins = doc.get("pins") or {}
    print("\npins 类型 %s，%d 个键" % (type(pins).__name__, len(pins)))
    if pins:
        pk = next(iter(pins))
        print("pins[%s] = %s" % (pk, json.dumps(pins[pk], ensure_ascii=False)[:800]))
    nets = doc.get("nets") or []
    print("\nnets 类型 %s，%d 条" % (type(nets).__name__, len(nets)))
    if nets:
        print("nets[0] = %s" % json.dumps(nets[0], ensure_ascii=False)[:800])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
