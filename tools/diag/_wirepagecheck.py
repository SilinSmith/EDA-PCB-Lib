# -*- coding: utf-8 -*-
"""_wirepagecheck.py —— 静态自检生成的导线页（不依赖浏览器）。"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT / "work" / "flow" / "导线workflow.html"

h = P.read_text(encoding="utf-8")
ok = True
print("文件 %s" % P)
print("  字节 %d ｜ UTF-8 无 BOM %s" % (len(h.encode("utf-8")),
                                   not h.startswith("\ufeff")))
for tag in ("graphdata", "docmd", "recipemd"):
    m = re.search(r'<script type="application/json" id="%s">(.*?)</script>' % tag, h, re.S)
    if not m:
        print("  [FAIL] 找不到 %s" % tag)
        ok = False
        continue
    try:
        d = json.loads(m.group(1).replace("<\\/", "</"))
    except Exception as e:                                          # noqa: BLE001
        print("  [FAIL] %s 不是合法 JSON：%s" % (tag, e))
        ok = False
        continue
    if tag == "graphdata":
        want = {"in": set(), "out": set()}
        for n in d["nodes"]:
            for p in n.get("ins") or []:
                want["in"].add("%s.%s" % (n["id"], p["id"]))
            for p in n.get("outs") or []:
                want["out"].add("%s.%s" % (n["id"], p["id"]))
        bad = [L for L in d["links"]
               if L["from"] not in want["out"] or L["to"] not in want["in"]]
        print("  graphdata: nodes %d links %d groups %d ｜ 悬空连线 %d"
              % (len(d["nodes"]), len(d["links"]), len(d.get("groups") or []), len(bad)))
        print("            EXTRA keys: %s" % sorted(set(d) - {"nodes", "links", "groups", "layout"}))
        ok = ok and not bad
    else:
        print("  %s: %d 字符、%d 行、含表格 %d 个、含代码块 %d 个"
              % (tag, len(d), d.count("\n") + 1, d.count("|---"), d.count("```") // 2))
        if len(d) < 200:
            print("  [FAIL] %s 太短，像是没灌进去" % tag)
            ok = False
# 内嵌 JS 抽出来给 node --check
js = re.findall(r"<script>(.*?)</script>", h, re.S)
if js:
    (ROOT / "work" / "_wirepage_check.js").write_text(max(js, key=len), encoding="utf-8")
    print("  js 已导出 → %s" % (ROOT / "work" / "_wirepage_check.js"))
print("结论：%s" % ("全部通过 ✓" if ok else "有问题 ✗"))
raise SystemExit(0 if ok else 1)
