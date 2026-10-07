# -*- coding: utf-8 -*-
"""把 `graph.json` 灌进 `template.html`，产出一个自包含的工作流编辑器。

    python work\\tools\\flow\\build.py

产出 `work\\flow\\workflow.html`（单文件，双击即可，不需要服务器）。

为什么内嵌而不是 fetch：浏览器打开 `file://` 页面时 `fetch('graph.json')`
会被 CORS 挡掉，所以 JSON 必须写进页面里。用户在页面上改完点「导出 JSON」
拿到新的 graph.json，再由 `apply.py`（我自己读）来落地。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # `Y:\workspace\EDA`
GRAPH = HERE / "graph.json"
TPL = HERE / "template.html"
OUT_DIR = ROOT / "work" / "flow"
OUT = OUT_DIR / "workflow.html"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    tpl = TPL.read_text(encoding="utf-8")

    # 只做最基本的转义：JSON 里不能出现裸的 `</script>`
    blob = json.dumps(g, ensure_ascii=False, indent=1).replace("</", "<\\/")
    html = tpl.replace("__GRAPH__", blob)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    n_n, n_l = len(g["nodes"]), len(g["links"])
    n_c = sum(1 for n in g["nodes"] if n.get("group") == "cand")
    print("写出 %s" % OUT)
    print("  节点 %d（其中候选 %d）  连线 %d  分组 %d"
          % (n_n, n_c, n_l, len(g["groups"])))
    print("  %d 字节" % OUT.stat().st_size)

    bad = check(g)
    if bad:
        print("!! 悬空连线 %d 条（端口不存在，画不出来）：" % len(bad))
        for s in bad:
            print("   " + s)
        return 1
    print("  自检：全部连线的端口都存在 ✓")
    return 0


def check(g: dict) -> list:
    """每条连线的两端必须指到真实存在的端口，否则 SVG 里画不出来。"""
    want = {"in": set(), "out": set()}
    for n in g["nodes"]:
        for p in n.get("ins") or []:
            want["in"].add("%s.%s" % (n["id"], p["id"]))
        for p in n.get("outs") or []:
            want["out"].add("%s.%s" % (n["id"], p["id"]))
    seen, bad = set(), []
    for L in g["links"]:
        f, t = L.get("from", ""), L.get("to", "")
        if f not in want["out"]:
            bad.append("from %s（没有这个输出端口）" % f)
        if t not in want["in"]:
            bad.append("to   %s（没有这个输入端口）" % t)
        key = (f, t)
        if key in seen:
            bad.append("重复连线 %s -> %s" % key)
        seen.add(key)
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
