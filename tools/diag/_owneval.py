"""同类相邻框：把「判据阈值」和「并了会不会误并真符号」放在同一张表上。

对 19 张快测集的 stage1 产物，取**同类型的框对**，量
    cd   = max(|dx|, |dy|)             两个框中心的切比雪夫距离（贴章后 = 官方 corner_dev）
    side = min(两框的 min(w,h))         自身尺寸（贴章后同类同尺寸）
    frac = cd / side
然后对每一对问一句：**它们的最近 GT（同类、corner_dev<=20）是不是同一颗？**
    same=True  ⇒ 这是真的重复检测，并掉是对的
    same=False ⇒ 两颗不同的真符号挨得近，并掉会**误伤**

用法：python work\\tools\\diag\\_owneval.py [--dir 目录] [case ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
from schematic_parser import data as D                          # noqa: E402
from schematic_parser.metrics import canon_type                 # noqa: E402

CASES = ["1", "2", "3", "4", "5", "7", "8", "9", "10",
         "0092", "0048", "0019", "0024", "0075", "0086",
         "0002", "0001", "0042", "0103"]
S1 = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs\stage1")
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "--dir":
        S1 = Path(args[i + 1])
cases = [a for a in args if not a.startswith("--") and a not in (str(S1),)] or CASES

FRACS = [0.2, 0.3, 0.5, 0.7, 1.0]
CDS = [8.0, 12.0, 20.0]
buckets = {f"frac<={f}": [0, 0] for f in FRACS}
buckets.update({f"cd<={c:g}": [0, 0] for c in CDS})
by_cls = defaultdict(lambda: [0, 0, 0])          # 类 -> [总对数, 同 GT, 异 GT]
pairs_all = []

for cid in cases:
    p = S1 / f"{cid}.stage1.json"
    if not p.exists():
        print(f"跳过 {cid}（没有 {p.name}）")
        continue
    doc = json.loads(p.read_text(encoding="utf-8"))
    comps = doc.get("components") or []
    case = D.find_case(cid)
    gt = json.loads(Path(case.json_path).read_text(encoding="utf-8"))
    with Image.open(case.png_path) as im:
        W, H = im.size
    gts = []
    for k, rec in (gt.get("components") or {}).items():
        b = rec.get("bbox")
        if not b or len(b) != 4:
            continue
        gts.append((k, canon_type(rec.get("type")),
                    [b[0], H - b[3], b[2], H - b[1]]))

    def nearest_gts(box, t):
        """同类、corner_dev<=20 的 GT 键（可能不止一个）。"""
        out = []
        for k, gt_t, gb in gts:
            if gt_t != t:
                continue
            d = max(abs(box[i] - gb[i]) for i in range(4))
            if d <= 20.0:
                out.append((d, k))
        return out

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            ta, tb = canon_type(a.get("type")), canon_type(b.get("type"))
            if ta != tb:
                continue
            ba, bb = a.get("bbox_img"), b.get("bbox_img")
            if not ba or not bb:
                continue
            ca = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2)
            cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            cd = max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1]))
            side = min(min(ba[2] - ba[0], ba[3] - ba[1]),
                       min(bb[2] - bb[0], bb[3] - bb[1]))
            if side <= 0:
                continue
            frac = cd / side
            ka = {k for _, k in nearest_gts(ba, ta)}
            kb = {k for _, k in nearest_gts(bb, tb)}
            same = bool(ka & kb)
            pairs_all.append((cid, ta, cd, side, frac, same,
                              a.get("id"), b.get("id"), sorted(ka), sorted(kb)))
            by_cls[ta][0] += 1
            by_cls[ta][1 if same else 2] += 1
            for f in FRACS:
                if frac <= f:
                    buckets[f"frac<={f}"][0 if same else 1] += 1
            for c in CDS:
                if cd <= c:
                    buckets[f"cd<={c:g}"][0 if same else 1] += 1

print(f"\n同类型相邻框对：{len(pairs_all)} 对（{len(cases)} 张）")
print(f"\n{'判据':<14}{'并掉的':>8}{'其中真重复':>11}{'误伤真符号':>11}{'误伤率':>8}")
for lbl, (s, d) in buckets.items():
    t = s + d
    print(f"{lbl:<14}{t:>8}{s:>11}{d:>11}{(d / t if t else 0):>7.1%}")

print(f"\n{'类':<16}{'对':>5}{'同GT':>7}{'异GT':>7}{'异GT率':>8}")
for k, (t, s, d) in sorted(by_cls.items(), key=lambda kv: -kv[1][0])[:14]:
    print(f"{k:<16}{t:>5}{s:>7}{d:>7}{(d / t if t else 0):>7.1%}")

print("\n最危险的（异 GT 且 frac 很小）前 15 对：")
for r in sorted([x for x in pairs_all if not x[5]], key=lambda x: x[4])[:15]:
    cid, t, cd, side, frac, same, ia, ib, ka, kb = r
    print(f"  {cid:>5} {t:<12} cd={cd:5.1f} side={side:5.1f} frac={frac:5.2f} "
          f"{ia}->{ka}  {ib}->{kb}")
