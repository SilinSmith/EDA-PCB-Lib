# -*- coding: utf-8 -*-
"""对比 VLM 补强文档与融合层产物：哪些键被改了、改成什么。

    python work\tools\diag\_fusiondiff.py 0001 [0005 ...]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
VLM = Path(os.environ.get("FUSION_VLM_DIR", str(ROOT / "work" / "_fusion_vlm")))
OUT = Path(os.environ.get("FUSION_OUT_DIR", str(ROOT / "work" / "_fusion_out9p")))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def bb(c):
    b = c.get("bbox") or [0, 0, 0, 0]
    return tuple(round(float(x), 1) for x in b)


def main(argv):
    for case in argv or ["0001"]:
        a = load(VLM / ("%s.json" % case))
        b = load(OUT / ("%s.json" % case))
        ca, cb = a.get("components") or {}, b.get("components") or {}
        bybb_a = {}
        for k, v in ca.items():
            bybb_a.setdefault(bb(v), []).append(k)
        bybb_b = {}
        for k, v in cb.items():
            bybb_b.setdefault(bb(v), []).append(k)
        print("== %s  vlm %d 颗 / fusion %d 颗" % (case, len(ca), len(cb)))
        changed = 0
        for bx in sorted(set(bybb_a) | set(bybb_b)):
            ka = bybb_a.get(bx, [])
            kb = bybb_b.get(bx, [])
            if ka != kb:
                changed += 1
                if changed <= 25:
                    print("   %-30s %s -> %s" % (str(bx), ka, kb))
        print("   （按键/框比对：%d 颗的键变了）" % changed)
        na = sum(1 for k, v in ca.items() if (v or {}).get("value"))
        nb = sum(1 for k, v in cb.items() if (v or {}).get("value"))
        print("   带 value 的：%d -> %d" % (na, nb))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
