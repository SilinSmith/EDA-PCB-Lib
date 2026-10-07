# -*- coding: utf-8 -*-
"""形状闸落地前后的键对照（只读）。

用法：
    python work\\tools\\diag\\_shapechk.py [闸后目录] [基线目录]
默认：闸后 `work/_shape_out`，基线 `work/_fusion_out`。
"""
import glob
import json
import os
import sys

WATCH = ("U580", "U1", "L44", "R1210", "R110", "USB", "U0", "p001")


def keys_of(d, case):
    p = os.path.join(d, case + ".json")
    if not os.path.exists(p):
        return None
    doc = json.load(open(p, encoding="utf-8"))
    return list((doc.get("components") or {}).keys())


def main(after, base):
    cases = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(after, "*.json"))
                   if not os.path.basename(p).startswith("_"))
    print("=== 闸后 %s（%d 张）" % (after, len(cases)))
    for c in cases:
        ka = keys_of(after, c)
        kb = keys_of(base, c)
        hit_a = [k for k in ka if k in WATCH]
        hit_b = [k for k in kb if k in WATCH] if kb is not None else None
        print("  %s  n=%d  关注键=%s" % (c, len(ka), hit_a))
        if kb is not None and hit_b != hit_a:
            print("       基线 n=%d 关注键=%s" % (len(kb), hit_b))
    # 逐 cid 看被闸掉的候选
    print("=== shape_gate 笔记")
    for c in cases:
        p = os.path.join(after, "_fusion_report.json")
        break
    rp = os.path.join(after, "_fusion_report.json")
    if os.path.exists(rp):
        rep = json.load(open(rp, encoding="utf-8"))
        for r in (rep.get("cases") or []):
            cf = r.get("cropfix") or {}
            print("  %s cf=%s judge=%s audit=%s" % (
                r.get("case"), cf, (r.get("judge") or {}).get("q"),
                (r.get("audit") or {}).get("n")))


if __name__ == "__main__":
    A = sys.argv[1] if len(sys.argv) > 1 else "work/_shape_out"
    B = sys.argv[2] if len(sys.argv) > 2 else "work/_fusion_out"
    main(A, B)
