# -*- coding: utf-8 -*-
"""_wirekey.py —— 页面关键数字在位自检。"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
h = (ROOT / "work" / "flow" / "导线workflow.html").read_text(encoding="utf-8")
NEED = ["148", "239", "316", "346", "207", "0.2426", "0.1876", "0.1900", "0.3068",
        "0.4056", "63.2", "92", "27", "0.45", "0048", "desig_slots", "match_nets",
        "line_tol", "EDGE_SNAP", "WIRE_WELD", "WIRE_PERP", "pincol", "1730", "4777",
        "STAGE1_WIRE_OVER", "546", "MISSING"]
for k in NEED:
    print("  %-18s %s" % (k, "在" if k in h else "**缺**"))
i = h.find("const X = {")
print("const X 起点 %d，样例 %s" % (i, h[i:i + 90]))
print("曲线点容量：tol %d 个 / tp %d 个"
      % (h.count("curves") and 1, 1))
print("id 列表：", re.findall(r'id="([a-z]+)"', h)[:12])
