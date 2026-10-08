# -*- coding: utf-8 -*-
"""_wiretok.py —— 定位内嵌 JSON 的断点（一次性诊断）。"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
h = (ROOT / "work" / "flow" / "导线workflow.html").read_text(encoding="utf-8")
m = re.search(r'<script type="application/json" id="recipemd">(.*?)</script>', h, re.S)
s = m.group(1)
print("len", len(s))
i = 2290
print(repr(s[i - 120:i + 120]))
print("--- 0x3c 后 30 字符处:")
for j, ch in enumerate(s):
    if ch == "<":
        print(j, repr(s[j:j + 24]))
