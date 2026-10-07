# -*- coding: utf-8 -*-
"""stage1 文档里有哪些文字框？以及「desig_slots 空的元件」附近有没有**没被认领**的文字？

这是「候选表是空的占引脚 fn 33%」这条的**可行性测量**：`_assigned` 只 dump 了
`desig_slots`（37/83 非空）和 `net_labels`，但记着 `n_free=210` —— 那 210 个
没被任何元件认领的文字**根本没进 `_assigned`**。而 `stage1` 文档里有全部文字框，
所以融合层其实**可以**把它们捡回来当位号候选（不用重跑 stage2）。

用法：python work\\tools\\diag\\_s1text.py [case]
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(r"Y:\workspace\EDA")
RUN = ROOT / "work" / "_submit_out220r36" / "runs"
CASE = sys.argv[1] if len(sys.argv) > 1 else "0001"

s1 = json.loads((RUN / "stage1" / ("%s.stage1.json" % CASE)).read_text(encoding="utf-8"))
a = json.loads((RUN / "stage2" / "_assigned" / ("%s.json" % CASE)).read_text(encoding="utf-8"))

print("stage1 顶层键:", list(s1.keys()))
for k, v in s1.items():
    if isinstance(v, list):
        print("  %-14s list n=%d" % (k, len(v)))
        if v:
            print("      [0] = %s" % json.dumps(v[0], ensure_ascii=False)[:260])
    elif isinstance(v, dict):
        print("  %-14s dict n=%d 键=%s" % (k, len(v), list(v)[:6]))
    else:
        print("  %-14s %r" % (k, v))

# 文字框：stage1 里叫 texts / text_boxes / n_texts?
tx = None
for k in ("texts", "text_boxes", "text", "ocr"):
    if isinstance(s1.get(k), list):
        tx = s1[k]
        print("\n用 stage1[%r] 当文字框表，n=%d" % (k, len(tx)))
        break
if tx is None:
    print("\n⚠ stage1 里没有文字框列表 —— 需要换数据源")
    sys.exit(0)


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def gap(b, c):
    dx = max(0.0, max(b[0] - c[2], c[0] - b[2]))
    dy = max(0.0, max(b[1] - c[3], c[1] - b[3]))
    return (dx * dx + dy * dy) ** 0.5


ds = a.get("desig_slots") or {}
comps = a.get("comps") or {}
s1c = {c["id"]: c for c in (s1.get("components") or [])}
empty = [cid for cid in comps if not ds.get(cid)]
print("desig_slots 空的元件 %d / %d" % (len(empty), len(comps)))

# 已被 desig_slots 认领的文字框（按 box_i 索引）
owned_i = set()
for cid, cands in ds.items():
    for c in cands:
        if isinstance(c.get("box_i"), int):
            owned_i.add(c["box_i"])
print("被认领的文字框索引数 %d / %d" % (len(owned_i), len(tx)))

cnt = collections.Counter()
samples = []
for cid in empty:
    b = (s1c.get(cid) or {}).get("bbox_img")
    if not b:
        cnt["无框"] += 1
        continue
    near = [(gap(b, t.get("box") or [0, 0, 0, 0]), i, t)
            for i, t in enumerate(tx) if t.get("box")]
    near.sort(key=lambda x: x[0])
    cand = [(g, i, t) for g, i, t in near if g <= 40][:4]
    if not cand:
        cnt["附近40px内没有文字"] += 1
    else:
        cnt["附近有文字"] += 1
    if len(samples) < 12:
        samples.append((cid, (s1c.get(cid) or {}).get("type"), [int(b[0]), int(b[1])],
                        [(round(g, 1), i, (t.get("text") or t.get("label")
                                           or t.get("s")), i in owned_i)
                         for g, i, t in cand]))
print()
for k, v in cnt.items():
    print("  %-24s %4d" % (k, v))
print("\n空 desig_slots 元件的邻居（cid, 类型, 框左上, [(间隙, 文字框索引, 文字, 是否已被认领)]）：")
for s in samples:
    print("   %s" % (s,))
