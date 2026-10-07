# -*- coding: utf-8 -*-
"""数一数 220 张里每个分项**爆 0 / 近 0** 的图有多少（用当前最佳产物）。

用法：python work\tools\diag\_zerocount.py [score.json]
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"Y:\workspace\EDA")
P = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work/_fusion_out220nl2/_score.json"
d = json.loads(P.read_text(encoding="utf-8"))
cases = d["cases"]
print("产物:", P)
print("汇总:", json.dumps(d["summary"], ensure_ascii=False)[:400])
print("case 字段:", sorted(cases[0].keys()))
print()
KEYS = [("component_f1", "元件"), ("pin_f1", "引脚"),
        ("net_hypergraph_f1", "网络"), ("net_line_f1", "导线")]
n = len(cases)
for k, cn in KEYS:
    vals = [float(c.get(k) or 0.0) for c in cases]
    z = [c["case_id"] for c, v in zip(cases, vals) if v == 0.0]
    z2 = [c["case_id"] for c, v in zip(cases, vals) if 0.0 < v < 0.05]
    print("%s(%s)  F1=0: %d/%d (%.3f)   0<F1<0.05: %d   "
          "min %.4f  中位 %.4f  max %.4f"
          % (cn, k, len(z), n, len(z) / n, len(z2),
             min(vals), sorted(vals)[n // 2], max(vals)))
    if z:
        print("   爆0: %s" % " ".join(z))
    if z2:
        print("   近0: %s" % " ".join(z2))
    print()
# 总分也为 0 的图
tot = [float(c.get("score") or 0.0) for c in cases]
zt = [c["case_id"] for c, v in zip(cases, tot) if v < 1.0]
print("总分 < 1 分的图: %d %s" % (len(zt), " ".join(zt)))
# 元件 tp 也数一下
ctp = [(c["case_id"], (c.get("counts") or {})) for c in cases[:1]]
print("counts 样例:", json.dumps(ctp[0][1], ensure_ascii=False)[:300])
