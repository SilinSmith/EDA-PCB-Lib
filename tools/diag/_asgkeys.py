# -*- coding: utf-8 -*-
"""`_assigned/<case>.json` 的结构探针：有没有「没被任何元件认领的文字」。

目的：`_desigcand2.py` 量到「候选表是空的」占引脚 fn 的 33%。要判这是
**检测漏了**还是**关联没关联上**，就得看 `_assigned` 里有没有 free 的文字框
落在元件附近。

用法：python work\\tools\\diag\\_asgkeys.py [case]
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
RUN = ROOT / "work" / "_submit_out220r36" / "runs"
CASE = sys.argv[1] if len(sys.argv) > 1 else "0001"

a = json.loads((RUN / "stage2" / "_assigned" / ("%s.json" % CASE))
               .read_text(encoding="utf-8"))
print("顶层键:", list(a.keys()))
for k, v in a.items():
    if isinstance(v, dict):
        print("  %-16s dict  n=%d  样例键=%s" % (k, len(v), list(v)[:4]))
        k0 = next(iter(v), None)
        if k0 is not None:
            print("      %s -> %s" % (k0, json.dumps(v[k0], ensure_ascii=False)[:300]))
    elif isinstance(v, list):
        print("  %-16s list  n=%d" % (k, len(v)))
        if v:
            print("      [0] = %s" % json.dumps(v[0], ensure_ascii=False)[:300])
    else:
        print("  %-16s %r" % (k, v))

# 统计：有多少元件有 desig_slots / 多少个没被认领的文字
ds = a.get("desig_slots") or {}
comps = a.get("comps") or {}
n_ds = sum(1 for c in comps if ds.get(c))
print("\n元件 %d，其中 desig_slots 非空 %d，空 %d" % (len(comps), n_ds, len(comps) - n_ds))
for k in ("n_free", "n_owned", "n_texts", "struct_slots", "texts", "free"):
    if k in a:
        print("  %s = %s" % (k, json.dumps(a[k], ensure_ascii=False)[:200]))
# 有没有一个明确的「文字候选池」
for k, v in a.items():
    if "text" in k.lower() or "slot" in k.lower():
        if isinstance(v, list):
            print("  ★ %s: list n=%d 例 %s" % (
                k, len(v), json.dumps(v[:2], ensure_ascii=False)[:400]))
        elif isinstance(v, dict):
            c = collections.Counter(str(x.get("owner") or x.get("cid") or "-")
                                    for x in v.values() if isinstance(x, dict))
            print("  ★ %s: dict n=%d 归属分布(前5)=%s" % (k, len(v), c.most_common(5)))
