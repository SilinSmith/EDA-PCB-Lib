# -*- coding: utf-8 -*-
"""把 `own`（独家归属）节点并进**内容更全的那一版** graph，再写回 `graph.json`。

为什么需要这一步（教训，别再犯）
--------------------------------
`work\\flow\\workflow.html` 是 `build.py` 从 `tools\\flow\\graph.json` 生成的，但
两者**曾经脱节**：`graph.json` 停在 21 节点，而 html 一路长到 25 节点
（`ev_crop` / `cropfix` / `audit` / `boxscan` 四个节点 + 16 条连线，只在 html 里，
`graph.json` 中根本没有）。我直接跑了一次 `build.py`，于是把 25 节点那版**覆盖成了
22 节点** —— 幸好 `workflow.html.bak6` 还在。

**所以规矩是：重建之前先 `_recover.py` 比一下节点集合，别假设 graph.json 是最新的。**

这个脚本以 `graph_bak6.json`（25 节点，最新）为底，加上 `own` 节点并把
`ev_sym.det -> {ev_pin,ev_net,pool}.det` 改成经过 `own.det`，最后写回 `graph.json`
（原文件备份为 `graph_pre_own.json`）。

    python work\\tools\\flow\\_merge_own.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRAPH = HERE / "graph.json"
BASE = HERE / "graph_bak6.json"
BACKUP = HERE / "graph_pre_own.json"

OWN_NODE = {
    "id": "own",
    "group": "fz",
    "kind": "new",
    "title": "★★ 独家归属（一个东西只有一个主人）",
    "sub": "争同一个位置的主张者 → 只留一个；判负者降权并入，不销毁",
    "x": 40,
    "y": 1040,
    "w": 300,
    "file": "cvparts/own.py",
    "ins": [{"id": "det", "label": "元件框+类型"}],
    "outs": [{"id": "det", "label": "去争后的框"},
             {"id": "ledger", "label": "归属账（留痕）"}],
    "params": [["rule", "cd<=12"], ["demote", "0.85"]],
}

EVSYM_APPEND = (
    "\n\n**★ 本节点的输出先过「独家归属」再往下走**（见 `own` 节点）：贴章之后同类框"
    "都是同一个正方形，两个中心差 10px 的 20×20 框 **IoU 只有 0.33**，IoU-NMS 抑制不掉"
    " —— 而官方 `metrics` 是**一对一**匹配，一个 GT 被两个预测够到就是 1 TP + 1 FP。"
    "19 张实测：`cd<=12` 并掉 2 对、**误伤 0**；官方那 20px 容差并掉 46 对、误伤 21"
    "（45.7%），**不能**当并框半径。"
)

# 这三条连线原来直接从 `ev_sym` 出发，现在改从 `own` 出发
REROUTE = ("ev_pin.det", "ev_net.det", "pool.det")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    base = json.loads(BASE.read_text(encoding="utf-8"))
    ids = [n["id"] for n in base["nodes"]]
    if "own" in ids:
        print("base 里已经有 own，直接重建即可"); return 0

    if GRAPH.exists():
        shutil.copy2(GRAPH, BACKUP)
        print("原 graph.json 备份到 %s" % BACKUP.name)

    base["nodes"].append(OWN_NODE)
    links, n_out, n_in = [], 0, 0
    for L in base["links"]:
        if L["from"] == "ev_sym.det" and L["to"] in REROUTE:
            links.append(dict(L, **{"from": "own.det"}))
            n_out += 1
        else:
            links.append(L)
    links.append({"from": "ev_sym.det", "to": "own.det", "kind": "data"})
    n_in += 1
    base["links"] = links

    for n in base["nodes"]:
        if n["id"] == "ev_sym" and "独家归属" not in (n.get("note") or ""):
            n["note"] = (n.get("note") or "") + EVSYM_APPEND

    GRAPH.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出 %s：nodes=%d links=%d（改道 %d 条，新增 %d 条）"
          % (GRAPH.name, len(base["nodes"]), len(links), n_out, n_in))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
