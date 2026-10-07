# -*- coding: utf-8 -*-
"""从旧 `workflow.html` 里把内嵌的 graph 抠出来，和 `graph.json` 逐节点对比。

背景：`work\\flow\\workflow.html` 一度是「改了 graph.json 之后重建」的产物，但
有几版是**直接从更丰富的 graph 建的**（含 `ev_crop` / `cropfix` / `audit` /
`boxscan` 四个节点），而 `graph.json` 里没有这四个。重建会把这部分内容抹掉。
这个脚本负责把差异摊开，供人工决定怎么合。

    python work\\tools\\flow\\_recover.py [html路径] [--out 输出json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRAPH = HERE / "graph.json"
PAT = re.compile(r'<script type="application/json" id="graphdata">(.*?)</script>', re.S)


def from_html(p: Path) -> dict:
    s = p.read_text(encoding="utf-8", errors="replace")
    m = PAT.search(s)
    if not m:
        raise SystemExit("没有 graphdata：%s" % p)
    return json.loads(m.group(1).replace("<\\/", "</"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]) if args else (ROOT / "work" / "flow" / "workflow.html.bak6")
    old = from_html(src)
    new = json.loads(GRAPH.read_text(encoding="utf-8"))

    oi = {n["id"]: n for n in old["nodes"]}
    ni = {n["id"]: n for n in new["nodes"]}
    print("旧版 %s：nodes=%d links=%d" % (src.name, len(oi), len(old["links"])))
    print("graph.json：nodes=%d links=%d" % (len(ni), len(new["links"])))
    print("\n只在旧版里（重建会抹掉）：%s" % sorted(set(oi) - set(ni)))
    print("只在 graph.json 里（新增）：%s" % sorted(set(ni) - set(oi)))
    print("\n同 id 但内容不同的节点：")
    for k in sorted(set(oi) & set(ni)):
        diff = [f for f in set(oi[k]) | set(ni[k])
                if json.dumps(oi[k].get(f), ensure_ascii=False, sort_keys=True)
                != json.dumps(ni[k].get(f), ensure_ascii=False, sort_keys=True)]
        if diff:
            print("  %-12s 字段不同：%s" % (k, sorted(diff)))
    ol = {(L["from"], L["to"]) for L in old["links"]}
    nl = {(L["from"], L["to"]) for L in new["links"]}
    print("\n只在旧版里的连线 %d 条：" % len(ol - nl))
    for a, b in sorted(ol - nl):
        print("   %s -> %s" % (a, b))
    print("只在 graph.json 里的连线 %d 条：" % len(nl - ol))
    for a, b in sorted(nl - ol):
        print("   %s -> %s" % (a, b))

    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n旧版 graph 写到 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
