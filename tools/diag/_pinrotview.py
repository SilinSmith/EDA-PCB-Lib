# -*- coding: utf-8 -*-
"""看 `_pinrotprobe.py` 存下的逐帧读串：按 (cid, pin, field) 归组打印 0°/cw/ccw 三帧。

用法：
    python -X utf8 eda-pcb\\tools\\diag\\_pinrotview.py [st_json]

默认读 `work\\_protorun_st.json`（`_pinrotprobe.py` 的输出）。这是排查「旋转重读把本来
读对的脚改坏 / 三个朝向谁更强」的第一手视图：每行是同一脚同一字段在三帧下的读串与 conf，
末尾的 `emit` 段是真正写回的值（含 before），`skip` 段是各道闸门的计数。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "work" / "_protorun_st.json")
st = json.loads(P.read_text(encoding="utf-8"))
raw = st.get("raw") or []
by = {}
for r in raw:
    by.setdefault((r["cid"], r["pin"], r["field"]), {})[r["rot"]] = (r["text"], r["conf"])
for k in sorted(by):
    cid, pin, field = k
    cells = by[k]
    line = " ".join("%s=%r(%.2f)" % (w, cells[w][0], cells[w][1])
                    for w in ("0", "cw", "ccw") if w in cells)
    print("%s p%-3d %-4s %s" % (cid, pin, field, line))
print("---- vote ----")
for v in st.get("vote") or []:
    print("形状 %s -> %s（名 %d / 号 %d；0° 帧 名 %d / 号 %d）"
          % (v["shape"], v["rot"], v["name"], v["num"], v["base"][0], v["base"][1]))
print("---- emit ----")
for e in st.get("emit") or []:
    print("%s p%-3d %-4s -> %-6r (%.2f, %s) before=%s"
          % (e["cid"], e["pin"], e["field"], e["text"], e["conf"], e["rot"], e["before"]))
print("skip", json.dumps(st.get("skip") or {}, ensure_ascii=False))
