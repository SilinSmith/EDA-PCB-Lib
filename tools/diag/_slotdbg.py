# -*- coding: utf-8 -*-
"""把一个位号槽位的原始证据全部摊开：desig_slots 原文、stage1 的文字框、本体框。

用法：python _slotdbg.py [cid] [case]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"Y:\workspace\EDA\eda-pcb")))
sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(r"Y:\workspace\EDA\work\_submit_out220r36\runs")
cid = sys.argv[1] if len(sys.argv) > 1 else "c0015"
case = sys.argv[2] if len(sys.argv) > 2 else "0001"

s1 = json.loads((RUN / "stage1" / ("%s.stage1.json" % case)).read_text(encoding="utf-8"))
a = json.loads((RUN / "stage2" / "_assigned" / ("%s.json" % case)).read_text(encoding="utf-8"))

comp = next((c for c in (s1.get("components") or []) if c.get("id") == cid), None)
print("== 元件 %s ==" % cid)
print("  type=%s  bbox_img=%s" % ((comp or {}).get("type"),
                                  [round(v) for v in ((comp or {}).get("bbox_img") or [])]))
print("  assigned.comps[%s] = %s" % (cid, json.dumps(a["comps"].get(cid), ensure_ascii=False)))
print("\n== desig_slots[%s] 原文 ==" % cid)
for i, d in enumerate(a.get("desig_slots", {}).get(cid) or []):
    print("  [%d] text=%-8s canon=%-8s conf=%.2f d=%.1f fit=%.2f box_i=%s box=%s"
          % (i, d.get("text"), d.get("canon"), d.get("conf", 0), d.get("d", 0),
             d.get("fit", 0), d.get("box_i"),
             [round(v) for v in (d.get("box") or [])]))

s1 = s1                                                    # 文案：stage1 的文字
texts = s1.get("texts") or []
print("\n== stage1 texts 共 %d 个，与本体框/候选框重叠的 ==" % len(texts))
boxes = [(i, t) for i, t in enumerate(texts)]
for i, d in enumerate(a.get("desig_slots", {}).get(cid) or []):
    bi = d.get("box_i")
    if bi is None or bi < 0 or bi >= len(texts):
        print("  box_i=%s 越界（texts 只有 %d 个）" % (bi, len(texts)))
        continue
    t = texts[bi]
    print("  box_i=%d  texts[%d] bbox=%s" % (bi, bi, [round(v) for v in t.get("bbox", [])]))
    reads = t.get("reads") or {}
    print("      reads=%s" % {k: v for k, v in list(reads.items())[:8]})
    print("      conf=%s wide=%s wratio=%s" % (t.get("conf"), t.get("wide"), t.get("wratio")))
