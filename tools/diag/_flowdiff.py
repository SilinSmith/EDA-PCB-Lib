# -*- coding: utf-8 -*-
"""对比 `work\\flow\\workflow.html*` 各自内嵌的 graph（看重新构建有没有弄丢内容）。

    python work\\tools\\diag\\_flowdiff.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

PAT = re.compile(r'<script type="application/json" id="graphdata">(.*?)</script>', re.S)


def load(path: str):
    s = open(path, encoding="utf-8", errors="replace").read()
    m = PAT.search(s)
    if not m:
        return None
    raw = m.group(1).replace("<\\/", "</")
    try:
        return json.loads(raw)
    except Exception as e:                                     # noqa: BLE001
        return {"_err": str(e)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    root = r"work\flow"
    files = sorted(glob.glob(os.path.join(root, "workflow.html*")))
    files.append(r"work\tools\flow\graph.json")
    base = None
    for p in files:
        if p.endswith(".json"):
            try:
                g = json.load(open(p, encoding="utf-8"))
            except Exception as e:                             # noqa: BLE001
                print("%-24s <json fail %s>" % (os.path.basename(p), e))
                continue
        else:
            g = load(p)
        if g is None:
            print("%-24s %8d B  <无 graphdata>" % (os.path.basename(p), os.path.getsize(p)))
            continue
        if "_err" in g:
            print("%-24s %8d B  <解析失败 %s>" % (os.path.basename(p), os.path.getsize(p), g["_err"]))
            continue
        ids = [n.get("id") for n in g.get("nodes", [])]
        print("%-24s %8d B  nodes=%-3d links=%-3d" % (
            os.path.basename(p), os.path.getsize(p), len(ids), len(g.get("links", []))))
        if base is None:
            base = set(ids)
        else:
            extra = [i for i in ids if i not in base]
            miss = sorted(base - set(ids))
            if extra:
                print("      仅此版有：%s" % extra)
            if miss:
                print("      此版没有：%s" % miss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
